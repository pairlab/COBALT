from controllers.controller import Controller
import numpy as np
from copy import deepcopy


class OSCController(Controller):
    def __init__(self, robot, config, env_idx):
        super(OSCController, self).__init__(
            robot=robot,
            config=config,
            env_idx=env_idx,
        )

        self.config = config

        self.num_robots = self.config.num_device

    def preprocess_user_info(self, user_index, user_info, device_id, current_rotation):
        """
        This function is used to preprocess the user info before
        into the teleop command.

        Args:
            user_index (int): user id to index into to identify the robot
            user_info (dict): dictionary of user commands

        Return:
            user_info (dict): dictionary of user commands
        """
        teleop_command = self.get_teleop_command_from_user_info(user_index, user_info)

        control_dict = self.get_teleop_control_from_command(
            user_index, teleop_command, user_info, current_rotation=current_rotation
        )

        # Used in data collection - stores teleop commands from device
        self.teleop_commands[device_id] = deepcopy(teleop_command)
        self.teleop_controls[device_id] = deepcopy(control_dict)

        return control_dict

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

        Return:
            teleop_command (dict): dictionary with command keys
                position: relative / absolute position desired for robot eef
                rotation: absolute rotation desired for robot eef
                "gripper": gripper position width command for robot eef
        """

        # position translation
        dpos = user_info["dpos"]

        # transform absolute rotation from phone to align with robot frame
        rotation = user_info["rotation"]
        rotation = self.robot.transform_user_rotation_to_align_with_robot(rotation)

        # read grasp action
        grasp = user_info["grasp"]
        gripper_command = self.robot.get_gripper_command(grasp=grasp)

        dpos_to_use = np.array(dpos)

        teleop_command = {
            "position": dpos_to_use,
            "rotation": rotation,
            "gripper": gripper_command,
        }

        return teleop_command

    def get_teleop_control_from_command(
        self, user_index, teleop_command, user_info, current_rotation
    ):
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
            current_rotation (numpy array): current rotation of the robot end effector
        Return:
            teleop_control (dict): dictionary which contains low-level controls for the robot:
                position: OSC delta position control
                rotation: OSC delta euler rotation control
                gripper: gripper control
        """

        # do some scaling and conversions from user commands to robot commands
        position_control = self.robot.teleop_to_position_control(
            user_info["engaged"],
            user_index,
            dpos=teleop_command["position"],
            rotation=teleop_command["rotation"],
        )
        rotation_control = self.robot.teleop_to_rotation_control(
            user_info["engaged"],
            user_index,
            dpos=teleop_command["position"],
            rotation=teleop_command["rotation"],
            current_rotation=current_rotation,
        )
        return {
            "position": position_control,
            "rotation": rotation_control,
            "gripper": teleop_command["gripper"],
        }

    def _start_trial(self):
        """
        This function is called at the start of a teleoperation trial.
        """
        pass

    def _end_trial(self):
        """
        This function is called at the end of a teleoperation trial.
        """
        pass

    def close(self):
        """
        This function is called at the end of a teleoperation session.
        """
        pass
