"""
Robosuite robot interface for teleoperation.
Focuses on robot-specific transformations, observations, and control methods.
Environment management is handled by RobosuiteSimulator.
"""

import os
import numpy as np
from copy import deepcopy

import utils as U

try:
    import robosuite.utils.transform_utils as T
except ImportError:
    print("WARNING: could not import robosuite transform utils")
    T = None

from os.path import join as pjoin
from robots import TeleopRobot, IKRobot, OSCRobot


class RobosuiteRobot(TeleopRobot, IKRobot, OSCRobot):
    """
    Robot interface for Robosuite environments.
    Handles action/observation transformations and robot-specific utilities.
    """

    def __init__(self, config):
        """Initialize robot interface."""
        self.config = config

        # Robot configuration
        self.robot_names = deepcopy(list(self.config.robot.names.values()))
        self.task_name = self.config.robot.task.name
        self.use_eef_ctrl = self.config.controller.flag.osc
        self.num_robots = len(self.robot_names)

        # Control configuration
        self.gripper_controls = [[-1.0], [1.0]]  # open, closed

        # Setup position-only control flag
        if self.use_eef_ctrl:
            self.eef_position_only_ctrl = False

    def path_to_ik_urdf(self):
        """
        Returns the path to the urdf file used for doing inverse kinematics.

        :return ik_urdf: string, path to a urdf
        """
        if self.robot_names[0] == "Panda":
            return pjoin(os.path.dirname(os.path.realpath(__file__)), "../assets/panda_description/urdf/panda_arm.urdf")
        else:
            raise Exception("Invalid robot.")

    def ik_joint(self):
        """
        Joint number to do IK computations on.

        :return ik_joint: a joint number
        """
        if self.robot_names[0] == "Panda":
            return 7
        else:
            raise Exception("Invalid robot.")

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

        If @eef_link is True, the rotation will correspond to the last link of the
        arm (usually for PyBullet IK).
        """
        if self.robot_names[0] in ["Panda"]:
            if eef_link:
                # post multiply by rotation of panda eef in eef, since
                # the rotation we receive corresponds to target eef rotation
                # in base frame
                rotation_correction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
                rotation_correction = rotation_correction.dot(np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]))

                # this corrects for link 8 being rotated 45 degrees from the end effector frame
                rotation_correction = rotation_correction.dot(U.rotation_matrix(angle=np.pi / 4, direction=[0.0, 0.0, 1.0], point=None)[:3, :3])

                # we are rotating the initial configuration for easy stacking
                rotation_correction = rotation_correction.dot(U.rotation_matrix(angle=-np.pi / 2, direction=[0.0, 0.0, 1.0], point=None)[:3, :3])

                rotation = rotation.dot(rotation_correction)
            else:
                # post multiply by rotation of panda eef in eef, since
                # the rotation we receive corresponds to target eef rotation
                # in base frame
                rotation_correction = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

                # note: this correction was introduced for the default panda pose in robosuite v1
                rotation_correction = rotation_correction.dot(U.rotation_matrix(angle=np.pi / 2, direction=[0.0, 0.0, 1.0], point=None)[:3, :3])

                rotation = rotation.dot(rotation_correction)
            return rotation
        else:
            raise Exception("Invalid robot.")

    ### Robot Control Methods ###

    def teleop_to_position_control(self, engaged, user_index, dpos, rotation, absolute=False):
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

        if (
            engaged
            and "Two" in self.config.task
            and "single-arm-opposed" in self.config.env_configuration
        ):

            initial = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pose = np.identity(4)
            pose[:3, 3] = dpos
            pose[:3, :3] = rotation

            if user_index == 0:
                z_90_pose = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                pose = z_90_pose.T @ pose @ initial.T @ z_90_pose @ initial
                dpos = pose[:3, 3]

            if user_index == 1:
                z_90_pose = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                pose = z_90_pose.T @ pose @ initial.T @ z_90_pose @ initial
                dpos = pose[:3, 3]

        return dpos * 100.0

    def teleop_to_rotation_control(self, engaged, user_index, dpos, rotation, current_rotation, absolute=False):
        """
        Converts teleoperation position displacement (@dpos) and
        an absolute teleoperation rotation (@rotation) to a
        delta rotation control command for the robot.
        """

        if absolute:
            raise Exception("Not Implemented.")

        if self.eef_position_only_ctrl:
            return None

        if not engaged:
            return np.zeros(3)  # delta rotation control

        if (
            engaged
            and "Two" in self.config.task
            and "single-arm-opposed" in self.config.env_configuration
        ):

            initial = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            pose = np.identity(4)
            pose[:3, 3] = dpos
            pose[:3, :3] = rotation

            if user_index == 0:
                z_90_pose = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                pose = z_90_pose.T @ pose @ initial.T @ z_90_pose @ initial
                rotation = pose[:3, :3]

            if user_index == 1:
                z_90_pose = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                pose = z_90_pose.T @ pose @ initial.T @ z_90_pose @ initial
                rotation = pose[:3, :3]

        # OSC delta rotation convention changed in version 1.2
        if T is not None:
            delta_rot_mat = rotation.dot(current_rotation.T)
            delta_quat = T.mat2quat(delta_rot_mat)
            return T.quat2axisangle(delta_quat)
        else:
            # Fallback if robosuite transform utils not available
            return np.zeros(3)
