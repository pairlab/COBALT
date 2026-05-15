"""
Abstract base class for any robot that can be controlled with
inverse kinematics. All abstract methods must be implemented.
"""

import abc  # for abstract base class definitions
import six  # preserve metaclass compatibility between python 2 and 3


@six.add_metaclass(abc.ABCMeta)
class IKRobot:
    """
    Abstract interface to be implemented for each real and simulated robot
    that wants to be controlled with inverse kinematics.

    Note: IK control is currently not supported in this release.
    """

    def __init__(self):
        raise NotImplementedError

    ### Config Methods ###
    @abc.abstractmethod
    def path_to_ik_urdf(self):
        """
        Returns the path to the urdf file used for doing inverse kinematics.

        :return ik_urdf: string, path to a urdf
        """
        raise NotImplementedError

    @abc.abstractmethod
    def ik_joint(self):
        """
        Joint number to do IK computations on.

        :return ik_joint: a joint number
        """
        raise NotImplementedError
