"""
Interface to the keyboard.

NOTE: Whenever using keyboard interface, you must run the client script as root.
"""

import numpy as np
from pynput.keyboard import Key, Listener
from configs import BaseClientConfig as Config
import utils as U
import logging


class KeyboardClient(object):

    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger('TeleoperationServer')

        self._display_controls()
        self.reset_internal_state()

        # reset signal
        self.reset_state = 0

        # task completion acknowledgment, we will toggle this
        self.task_completion_ack = 0
        self.task_timeout_ack = 0

        if self.config.client.first_person:
            self.first_person_transform = U.make_pose(np.zeros(3), np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))[:3, :3]

        # make a thread to listen to keyboard and register our callback functions
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print(f"{char}\t{info}")

        print("")
        print_command("key", "command")
        print_command("e", "enable/disable control")
        print_command("q", "reset the simulation")
        print_command("spacebar", "open/close the gripper")
        print_command("w-a-s-d", "move arm in a horizontal plane")
        print_command("r-f", "move arm vertically")
        print_command("z-x", "rotate arm along x-axis")
        print_command("t-g", "rotate arm along y-axis")
        print_command("c-v", "rotate arm along z-axis")

    def reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """

        self.engage = False

        # we will send desired position and orientation (same as VR interface)
        self.pos = np.zeros(3)  # (x, y, z)
        self.grasp = False
        self.last_pos = np.zeros(3)

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

        self.logger.debug('Keyboard client internal states reset')

    def get_controller_state(self):
        """
        Returns the current keyboard state, a dictionary of pos, orn, and grasp.
        """

        if self.config.client.first_person:
            pos = self.first_person_transform.dot(self.pos)
            rotation = self.first_person_transform.dot(self.rotation.dot(self.first_person_transform))
        else:
            pos = self.pos
            rotation = self.rotation

        dpos = pos - self.last_pos
        self.last_pos = np.array(pos)

        state = dict(
            dpos=list(dpos),
            rotation=list(rotation.reshape(-1)),
            grasp=self.grasp,
            engage=self.engage,
            reset=self.reset_state,
            completion=self.task_completion_ack,
            timeout=self.task_timeout_ack
        )
        self.logger.debug(f"Keyboard controller state: {state}")

        return state

    def on_press(self, key):
        """
        Key handler for key presses.
        """

        print(f"{key} pressed")

        # note that some keys don't have the "char" attribute
        # this might lead to an AttributeError
        try:
            # controls for moving position
            if key.char == "w":
                self.pos[0] -= self.config.keyboard.delta_pos  # dec x
            elif key.char == "s":
                self.pos[0] += self.config.keyboard.delta_pos  # inc x
            elif key.char == "a":
                self.pos[1] -= self.config.keyboard.delta_pos  # dec y
            elif key.char == "d":
                self.pos[1] += self.config.keyboard.delta_pos  # inc y
            elif key.char == "f":
                self.pos[2] -= self.config.keyboard.delta_pos  # dec z
            elif key.char == "r":
                self.pos[2] += self.config.keyboard.delta_pos  # inc z

            # controls for moving orientation
            elif key.char == "z":
                drot = U.rotation_matrix(angle=0.1, direction=[1.0, 0.0, 0.0], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
            elif key.char == "x":
                drot = U.rotation_matrix(angle=-0.1, direction=[1.0, 0.0, 0.0], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
            elif key.char == "t":
                drot = U.rotation_matrix(angle=0.1, direction=[0.0, 1.0, 0.0], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
            elif key.char == "g":
                drot = U.rotation_matrix(angle=-0.1, direction=[0.0, 1.0, 0.0], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
            elif key.char == "c":
                drot = U.rotation_matrix(angle=0.1, direction=[0.0, 0.0, 1.0], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
            elif key.char == "v":
                drot = U.rotation_matrix(angle=-0.1, direction=[0.0, 0.0, 1.0], point=None)[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z

        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        """

        try:
            # controls for grasping
            if key == Key.space:
                self.grasp = not self.grasp  # toggle gripper

            # control for toggling the engage button
            elif key.char == "e":
                self.engage = not self.engage  # toggle engage

            # user-commanded reset
            elif key.char == "q":
                self.reset_state = 1
                self.reset_internal_state()

        except AttributeError as e:
            pass

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

        self.logger.info("You completed the task!")
        self.reset_internal_state()

        # toggle this bit to deliver an acknowledgment to the server
        self.task_completion_ack = (self.task_completion_ack + 1) % 2

    def handle_task_timeout(self):
        """
        Called on task timeout. Should disengage control and handle this appropriately.
        """

        self.logger.info("TIMEOUT: You took too long to complete the task...")
        self.reset_internal_state()

        # toggle this bit to deliver an acknowledgment to the server
        self.task_timeout_ack = (self.task_timeout_ack + 1) % 2

    def stop(self):
        # stop listening
        self.listener.stop()
        self.logger.info('Keyboard client stopped')


if __name__ == '__main__':
    KC = KeyboardClient()
    import time

    while True:
        print(KC.get_controller_state())
        time.sleep(0.1)
