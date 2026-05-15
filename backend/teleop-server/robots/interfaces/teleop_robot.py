"""
Abstract base class for any robot that can be teleoperated.
All abstract methods must be implemented.
"""

import abc  # for abstract base class definitions
import six  # preserve metaclass compatibility between python 2 and 3

# global dictionary for remembering name - class mappings
REGISTERED_ROBOTS = {}


def register_robot(target_class):
    """
    Registers a robot class in the global registry.

    :param target_class: the class to register
    """
    REGISTERED_ROBOTS[target_class.__name__] = target_class


def make_robot(robot_name, *args, **kwargs):
    """
    Creates an instance of a robot. Make sure to pass any other needed arguments.

    :param robot_name: name for the robot class
    """
    if robot_name not in REGISTERED_ROBOTS:
        raise Exception(f"Robot {robot_name} not found. Make sure it is a registered robot among: {', '.join(REGISTERED_ROBOTS)}")
    return REGISTERED_ROBOTS[robot_name](*args, **kwargs)


class TeleopRobotMeta(abc.ABCMeta):
    """
    Define a metaclass for constructing an abstract base class.
    It also registers robot classes into the global registry.
    """

    def __new__(meta, name, bases, class_dict):
        cls = super(TeleopRobotMeta, meta).__new__(meta, name, bases, class_dict)
        register_robot(cls)
        print(f"Registering Robot {cls}")
        return cls


@six.add_metaclass(TeleopRobotMeta)
class TeleopRobot(object):
    """
    A general teleoperation robot base class wrapper for all simulation robots
    to conform to.
    """

    def __init__(self):
        pass

    @property
    def name(self):
        """
        Returns name of the robot. This defaults to the class name
        """
        return self.__class__.__name__

    ### Robot Observation Methods ###

    @abc.abstractmethod
    def get_gripper_command(self, grasp):
        """
        Returns a continuous raw control signal for the gripper based
        on the boolean @grasp value - if True, return the signal
        corresponding to closing the gripper, and if False, return
        the signal corresponding to opening the gripper.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transform_user_rotation_to_align_with_robot(self, rotation, eef_link=False):
        """
        Input rotation matrices from the user teleop device were designed
        with a particular end effector in mind, but other robots have different
        conventions for the end effector axis with respect to the base frame.
        Each robot must implement its own modification to the user rotation
        command here to account for this.

        If @eef_link is True, the rotation will correspond to the last link of the
        arm (usually for PyBullet IK).
        """
        raise NotImplementedError

    def reset_user_teleop_state(self, user_index):
        """
        This function is used to reset any state associated with a user when they disengage from controlling the robot.
        This is important to prevent sudden jumps in the robot position when a user engages after a period of disengagement.
        Currently, this function is used when we need to compute delta rotations from the user teleoperation device, as we currently receive absolute rotations.

        Args:
            user_index (int): user id to index into to identify the robot
        """
        pass
