"""
Abstract base class for all robot controllers.
All abstract methods must be implemented.
"""

import abc  # for abstract base class definitions
import six  # preserve metaclass compatibility between python 2 and 3
import time
import cv2

# global dictionary for remembering name - class mappings
REGISTERED_CONTROLLERS = {}


def register_controller(target_class):
    """
    Registers a controller class in the global registry.

    :param target_class: the class to register
    """
    REGISTERED_CONTROLLERS[target_class.__name__] = target_class


def make_controller(controller_name, *args, **kwargs):
    """
    Creates an instance of a controller. Make sure to pass any other needed arguments.

    :param controller_name: name for the robot class
    """
    if controller_name not in REGISTERED_CONTROLLERS:
        raise Exception(f"Controller {controller_name} not found. Make sure it is a registered controller among: {', '.join(REGISTERED_CONTROLLERS)}")
    return REGISTERED_CONTROLLERS[controller_name](*args, **kwargs)


class ControllerMeta(abc.ABCMeta):
    """
    Define a metaclass for constructing an abstract base class.
    It also registers controller classes into the global registry.
    """

    def __new__(meta, name, bases, class_dict):
        cls = super(ControllerMeta, meta).__new__(meta, name, bases, class_dict)
        register_controller(cls)
        return cls


@six.add_metaclass(ControllerMeta)
class Controller(object):
    """
    Base class for all robot controllers.
    Defines basic interface for all controllers to adhere to.
    """

    def __init__(
        self,
        robot,
        config,
        env_idx,
    ):
        self.robot = robot
        self.config = config
        self.env_idx = env_idx

    def start_trial(self):
        """
        This function is called at the start of a teleoperation trial.
        """
        self.num_robots = self.config.num_device

        # initialize some state variables
        self.counter = 0  # counts
        self.teleop_commands = {}
        self.teleop_controls = {}

        # subclass function
        self._start_trial()

    def end_trial(self):
        """
        This function is called at the end of a teleoperation trial.
        """

        # subclass function
        self._end_trial()

    def collect_session(
        self,
        user_infos_by_device,
        image_data,
    ):
        """
        Returns a dictionary of commands for data collection for an entire session
        (potentially multiple devices). Counter is incremented once per call.
        """
        self.counter += 1

        ret = {
            f"counter": self.counter,
            "timestamps/control_loop_time": time.time(),
        }

        for device_id, user_info in user_infos_by_device.items():
            ret[f"enabled/{device_id}"] = user_info["engaged"]
            ret[f"timestamps/sent/{device_id}"] = user_info["sent"]
            ret[f"timestamps/received/{device_id}"] = user_info["received"]

        if self.config.data_collection.enabled:
            for device_id in user_infos_by_device.keys():
                if device_id in self.teleop_commands:
                    ret[f"teleop_commands/{device_id}"] = self.teleop_commands[device_id]

            ret["robot_controls"] = {}
            for device_id in user_infos_by_device.keys():
                if device_id in self.teleop_controls:
                    ret["robot_controls"][f"action/{device_id}"] = self.teleop_controls[device_id]

            if self.config.store_image_data:
                resized_images = {}
                for view_name, view_image in image_data.items():
                    resized_images[view_name] = cv2.resize(
                        view_image,
                        (self.config.image_width, self.config.image_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                ret["image_data"] = resized_images

        controller_info = {
            "controller_info": ret,
            "env_idx": self.env_idx,
        }

        return controller_info

    @abc.abstractmethod
    def preprocess_user_info(self, user_index, user_info, device_id):
        """
        This function is used to preprocess the user info before
        into the teleop command.

        Args:
            user_index (int): user id to index into to identify the robot
            user_info (dict): dictionary of user commands

        Return:
            user_info (dict): dictionary of user commands
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_teleop_command_from_user_info(self, user_index, user_info):
        """
        This function is used to parse the dictionary of commands
        from the user and returns a teleoperation command
        dictionary to send to the robot.

        NOTE: Should return None if no valid control is available or if
              robot should not be controlled for some reason.

        Args:
            user_index (int): user id to index into to identify the robot
            user_info (dict): dictionary of user commands
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_teleop_control_from_command(self, user_index, teleop_command, user_info):
        """
        This function is used to convert a teleoperation command
        dictionary to a teleoperation control dictionary. At
        a high-level, this does any necessary re-scaling and
        conversions to change delta positions and absolute
        rotations from a teleoperation device to low-level
        robot control.

        Args:
            user_index (int): user id to index into to identify the robot
            teleop_command (dict): dictionary which contains (at least) these keys
                position: processed position command from teleoperation device
                rotation: processed absolute rotation matrix command from teleoperation device
                gripper: gripper command

            user_info (dict): dictionary of user commands

        Return:
            teleop_control (dict): dictionary which contains low-level controls for the robot. This
                will vary depending on the controller.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def _start_trial(self):
        """
        This function is called at the start of a teleoperation trial.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _end_trial(self):
        """
        This function is called at the end of a teleoperation trial.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        This function is called at the end of a teleoperation session.
        """
        raise NotImplementedError
