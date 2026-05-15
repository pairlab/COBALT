"""
IsaacLab robot interface for teleoperation.
Focuses on robot-specific transformations, observations, and control methods.
Environment management is handled by IsaacLabSimulator.
"""

import numpy as np
from copy import deepcopy

from robots import TeleopRobot, IKRobot, OSCRobot
import utils as U


class IsaacLabRobot(TeleopRobot, IKRobot, OSCRobot):
    """
    Wraps a Isaac Lab environment.
    """

    def __init__(self, config):
        self.config = config
        self.robot_names = deepcopy(list(self.config.robot.names.values()))
        self.num_robots = len(self.robot_names)

        self.eef_position_only_ctrl = False

        self.gripper_controls = [[1.0], [-1.0]]

    def path_to_ik_urdf(self):
        raise NotImplementedError("IK control is currently not supported.")

    def ik_joint(self):
        raise NotImplementedError("IK control is currently not supported.")

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
        if self.robot_names[0] in ["IsaacLabRobot"]:
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

        return dpos * 25.0

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

        # use the OSC controller's convention for delta rotation
        quat_mat = np.array([
            current_rotation[1],
            current_rotation[2],
            current_rotation[3],
            current_rotation[0],
        ])

        current_rotation = U.quat2mat(quat_mat)

        # OSC delta rotation convention changed in version 1.2
        delta_rot_mat = rotation.dot(current_rotation.T)

        delta_quat = U.mat2quat(delta_rot_mat)

        return U.quat2axisangle(delta_quat)
