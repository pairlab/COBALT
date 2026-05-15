# Mac driver for SpaceNav controller
from __future__ import print_function
import hid
import time
import threading
import numpy as np

from configs import BaseClientConfig as Config
import utils as U

# timing in seconds to register double click
DOUBLE_CLICK_TIME = 0.3


# convert two 8 bit bytes to a signed 16 bit integer
def to_int16(y1, y2):
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350.0):
    x = x / axis_scale
    x = min(max(x, -1.0), 1.0)
    return x


def convert(b1, b2):
    return scale_to_control(to_int16(b1, b2))


class SpaceNavigator(object):

    def __init__(
        self,
        vendor_id=9583,
        product_id=50741,  # wireless connection
    ):

        print("Opening SpaceNavigator device")
        self.device = hid.device()
        self.device.open(vendor_id, product_id)  # SpaceNavigator

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        # state variables to detect whether left or right buttons are held down as a single or double click
        self.single_left_click_and_hold = False
        self.double_left_click_and_hold = False
        self.single_right_click_and_hold = False
        self.double_right_click_and_hold = False

        self._control = [0, 0, 0, 0, 0, 0]

        self.config = Config()

        self.reset_internal_state()

        # reset signal
        self.reset_state = 0

        # task completion acknowledgment, we will toggle this
        self.task_completion_ack = 0
        self.task_timeout_ack = 0

        if self.config.client.first_person:
            self.first_person_transform = U.make_pose(np.zeros(3), np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))[:3, :3]

        # launch daemon thread to listen to SpaceNav
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """

        # whether spacemouse control is enabled
        self.engage = False

        # reset controls
        self._control = [0, 0, 0, 0, 0, 0]

        # we will send desired position and orientation
        self.grasp_state = False
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

    def get_controller_state(self):
        """
        Returns the current state of the 3d mouse, a dictionary of pos, orn, and grasp.
        """
        dpos = self.control[:3] * self.config.mouse.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * self.config.mouse.orn_sensitivity

        # convert RPY to an absolute orientation
        drot1 = U.rotation_matrix(angle=-pitch, direction=[1.0, 0.0, 0.0], point=None)[:3, :3]
        drot2 = U.rotation_matrix(angle=roll, direction=[0.0, 1.0, 0.0], point=None)[:3, :3]
        drot3 = U.rotation_matrix(angle=yaw, direction=[0.0, 0.0, 1.0], point=None)[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))
        if self.config.client.first_person:
            dpos = self.first_person_transform.dot(dpos)
            rotation = self.first_person_transform.dot(self.rotation.dot(self.first_person_transform))
        else:
            rotation = self.rotation

        htamp = self.single_left_click_and_hold and self.single_right_click_and_hold

        return dict(
            dpos=list(dpos),
            rotation=list(rotation.reshape(-1)),
            grasp=self.grasp_state,
            engage=self.engage,
            reset=self.reset_state,
            htamp=htamp,
            completion=self.task_completion_ack,
            timeout=self.task_timeout_ack
        )

    def set_reset_state(self, state):
        """
        Set the internal reset variable used to send a reset signal to the server.
        """
        assert state == 0 or state == 1
        self.reset_state = state

    def handle_task_completion(self):
        """
        Called on task completion. Should disengage control and handle this appropriately.
        """

        print("You completed the task!")
        self.reset_internal_state()

        # toggle this bit to deliver an acknowledgment to the server
        self.task_completion_ack = (self.task_completion_ack + 1) % 2

    def handle_task_timeout(self):
        """
        Called on task timeout. Should disengage control and handle this appropriately.
        """

        print("TIMEOUT: You took too long to complete the task...")
        self.reset_internal_state()

        # toggle this bit to deliver an acknowledgment to the server
        self.task_timeout_ack = (self.task_timeout_ack + 1) % 2

    def run(self):

        t_last_left_click = -1
        t_last_right_click = -1

        # current left and right click values
        cur_left_click = False
        cur_right_click = False

        # previous left and right click values (to detect first press or first release)
        prev_left_click = False
        prev_right_click = False
        roll = 0
        pitch = 0
        yaw = 0
        while True:
            d = self.device.read(7)
            if d is not None:

                if d[0] == 1:
                    # read spacemouse motion
                    y = convert(d[1], d[2])
                    x = convert(d[3], d[4])
                    z = convert(d[5], d[6]) * -1.0

                    if self.engage:
                        # write spacemouse motion to control
                        self._control = [x, y, z, roll, pitch, yaw]

                if d[0] == 2:
                    # read spacemouse motion
                    roll = convert(d[1], d[2]) * 1.0
                    pitch = convert(d[3], d[4]) * -1.0
                    yaw = convert(d[5], d[6]) * 1.0

                    if self.engage:
                        # write spacemouse motion to control
                        self._control = [x, y, z, roll, pitch, yaw]

                elif d[0] == 3:

                    if d[1] == 0:
                        # no buttons are pressed
                        cur_left_click = False
                        cur_right_click = False
                    elif d[1] == 1:
                        # only left button is pressed
                        cur_left_click = True
                        cur_right_click = False
                    elif d[1] == 2:
                        # only right button is pressed
                        cur_left_click = False
                        cur_right_click = True
                    elif d[1] == 3:
                        # both left and right button is pressed
                        cur_left_click = True
                        cur_right_click = True

                    # detect click as low to high transition
                    got_left_click = cur_left_click and (not prev_left_click)
                    if got_left_click:
                        # note: this variable is used for grasping signal
                        self.single_left_click_and_hold = True

                        # remember when we got this click and detect a double click
                        t_left_click = time.time()
                        elapsed_time = t_left_click - t_last_left_click
                        t_last_left_click = t_left_click
                        if elapsed_time < DOUBLE_CLICK_TIME:
                            self.double_left_click_and_hold = True

                    got_right_click = cur_right_click and (not prev_right_click)
                    if got_right_click:
                        self.single_right_click_and_hold = True

                        # on right click, ensure we are enabled (we will get disabled by a reset or task completion)
                        self.engage = True

                        # remember when we got this click and detect a double click
                        t_right_click = time.time()
                        elapsed_time = t_right_click - t_last_right_click
                        t_last_right_click = t_right_click
                        if elapsed_time < DOUBLE_CLICK_TIME:
                            self.double_right_click_and_hold = True

                            # on right double click, we will trigger a reset
                            self.reset_state = 1
                            print("Got right click -- triggering reset...")
                            self.reset_internal_state()

                    # detect release as high to low transition
                    got_left_release = (not cur_left_click) and prev_left_click
                    if got_left_release:
                        self.single_left_click_and_hold = False
                        self.double_left_click_and_hold = False
                        # toggle grasp state on left click and release
                        self.grasp_state = not self.grasp_state

                    got_right_release = (not cur_right_click) and prev_right_click
                    if got_right_release:
                        self.single_right_click_and_hold = False
                        self.double_right_click_and_hold = False

                    # update prev clicks
                    prev_left_click = cur_left_click
                    prev_right_click = cur_right_click

    @property
    def control(self):
        """
        Returns 6-DoF control
        """
        return np.array(self._control)


if __name__ == '__main__':
    spacenav = SpaceNavigator()
    for i in range(1000):
        print(spacenav.control)
        time.sleep(0.02)
