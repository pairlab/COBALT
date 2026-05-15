"""
Robosuite robot interface for teleoperation.
Focuses on robot-specific transformations, observations, and control methods.
Environment management is handled by RobosuiteSimulator.
"""

import numpy as np
from copy import deepcopy

try:
    import robosuite.utils.transform_utils as T
except ImportError:
    print("WARNING: could not import robosuite transform utils")
    T = None

from robots import TeleopRobot, OSCRobot


class YAMRobot(TeleopRobot, OSCRobot):
    """
    Robot interface for Robosuite environments.
    Handles action/observation transformations and robot-specific utilities.
    """

    def __init__(self, config):
        """Initialize robot interface with existing environment."""
        self.config = config

        # Robot configuration
        self.robot_names = deepcopy(list(self.config.robot.names.values()))
        self.task_name = self.config.robot.task.name
        self.use_eef_ctrl = self.config.controller.flag.osc
        self.num_robots = len(self.robot_names)

        # Camera and control configuration
        self.gripper_controls = [[1.0], [0.0]]  # open, closed

        # State variables
        self._phone_init_rotation = [None for _ in range(self.num_robots)]
        self._robot_init_rotation = [None for _ in range(self.num_robots)]
        self._phone_to_robot_rotation_local = [None for _ in range(self.num_robots)]
        self._prev_phone_rotation = [None for _ in range(self.num_robots)]
        self._phone_to_robot_rotation_global = np.array(
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        )

        # Setup position-only control flag
        if self.use_eef_ctrl:
            self.eef_position_only_ctrl = False

    ### Robot Observation Methods ###

    def num_joints(self):
        """
        Returns the number of joints on the robot arm.

        :return num_joints: int
        """
        return 7

    def get_gripper_command(self, grasp):
        """
        Returns a continuous raw control signal for the gripper based
        on the boolean @grasp value - if True, return the signal
        corresponding to closing the gripper, and if False, return
        the signal corresponding to opening the gripper.
        """
        return self.gripper_controls[int(grasp)]

    def transform_user_rotation_to_align_with_robot(self, rotation, eef_link=False):
        """
        Input rotation matrices from the user teleop device were designed
        with a particular end effector in mind, but other robots have different
        conventions for the end effector axis with respect to the base frame.
        Each robot must implement its own modification to the user rotation
        command here to account for this.
        """
        return self._phone_to_robot_rotation_global @ rotation

    ### Robot Control Methods ###

    def teleop_to_position_control(
        self, engaged, user_index, dpos, rotation, absolute=False
    ):
        """
        Converts teleoperation position displacement (@dpos) and
        an absolute teleoperation rotation (@rotation) to a
        delta position control command for the robot. If @absolute
        is true, then both the position input and control output are
        an absolute position.
        """
        if absolute:
            # return absolute position command without any change
            return dpos

        if not engaged:
            return np.zeros(3)  # delta position control

        dpos = self._phone_to_robot_rotation_global @ dpos

        return dpos / 1.5

    def teleop_to_rotation_control(
        self, engaged, user_index, dpos, rotation, current_rotation, absolute=False
    ):
        """
        Converts teleoperation position displacement (@dpos) and
        an absolute teleoperation rotation (@rotation) to a
        delta rotation control command for the robot.
        """
        if absolute:
            raise Exception("Not Implemented.")

        if not engaged:
            return np.zeros(3)

        if self.eef_position_only_ctrl:
            return None

        if self._phone_to_robot_rotation_local[user_index] is None:
            self._phone_init_rotation[user_index] = rotation.copy()
            self._robot_init_rotation[user_index] = current_rotation.copy()
            self._phone_to_robot_rotation_local[user_index] = (
                self._phone_init_rotation[user_index].T
                @ self._robot_init_rotation[user_index]
            )

        phone_rotation = (rotation @ self._phone_to_robot_rotation_local[user_index])  # local transformation to align phone with robot eef

        prev_phone_rotation = self._prev_phone_rotation[user_index]
        if prev_phone_rotation is None:
            delta_rot_mat = np.eye(3)
        else:
            delta_rot_mat = prev_phone_rotation.T.dot(phone_rotation)
        self._prev_phone_rotation[user_index] = phone_rotation.copy()

        if T is not None:
            delta_quat = T.mat2quat(delta_rot_mat)
            return T.quat2axisangle(delta_quat)
        else:
            # Fallback if robosuite transform utils not available
            return np.zeros(3)

    def reset_user_teleop_state(self, user_index):
        self._phone_init_rotation[user_index] = None
        self._robot_init_rotation[user_index] = None
        self._phone_to_robot_rotation_local[user_index] = None
        self._prev_phone_rotation[user_index] = None
