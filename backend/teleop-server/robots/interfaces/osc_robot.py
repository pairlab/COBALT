"""
Abstract base class for any robot that can be controlled with
Operational Space Control (OSC).
All abstract methods must be implemented.
"""

import abc  # for abstract base class definitions
import six  # preserve metaclass compatibility between python 2 and 3


@six.add_metaclass(abc.ABCMeta)
class OSCRobot:
    """
    Abstract interface to be implemented for each real and simulated robot
    that wants to be controlled with Operational Space Control.
    """

    def __init__(self):
        raise NotImplementedError

    ### Robot Control Methods ###

    @abc.abstractmethod
    def teleop_to_position_control(self, dpos, rotation, absolute=False):
        """
        Converts teleoperation position displacement (@dpos) and
        an absolute teleoperation rotation (@rotation) to a
        delta position control command for the robot. If @absolute
        is true, then both the position input and control output are
        an absolute position.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def teleop_to_rotation_control(self, engaged, user_index, dpos, rotation):
        """
        Converts teleoperation position displacement (@dpos) and
        an absolute teleoperation rotation (@rotation) to a
        delta rotation control command for the robot.
        """
        raise NotImplementedError
