"""
Robosuite simulator implementation for the teleoperation server.
Handles environment setup, stepping, and resetting while keeping robot-specific
transformations in the separate robot interface.
"""

import os
import json
import numpy as np
from typing import Any, Dict, List, Tuple
import traceback
from collections.abc import Mapping
from copy import deepcopy

from .base_simulator import BaseSimulator
from utils.wrappers import StackSubprocVectorEnv, EnvStateWrapper, ResetWrapper, SuccessWrapper

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.wrappers import VisualizationWrapper

import platform
from os.path import join as pjoin

try:
    import mimicgen
except ImportError:
    print("WARNING: could not import mimicgen envs")

try:
    import robosuite_task_zoo
except ImportError:
    print("WARNING: could not import robosuite task zoo envs")

_SYSTEM = platform.system()
if _SYSTEM == "Windows":
    os.environ["MUJOCO_GL"] = "wgl"
elif _SYSTEM == "Darwin":
    os.environ["MUJOCO_GL"] = "cgl"
else:
    os.environ["MUJOCO_GL"] = "egl"


class RobosuiteSimulator(BaseSimulator):
    """Robosuite simulator implementation with LIBERO support."""

    def __init__(self, config: Any, **kwargs):
        super().__init__(config)

        # Environment setup variables
        self.env = None
        self.task_name = None
        self.robot_names = None
        self.use_rendering = False
        self.use_offscreen_rendering = True
        self.use_eef_ctrl = False

        # Task instance state variables
        self._current_task_instance_state = None
        self._current_task_instance_xml = None
        self._saved_task_instance_state = None
        self._saved_task_instance_xml = None

        # Cached robot state information
        self.cache_sim_state = None
        self.cache_robot_state = None
        self.cache_robot_eef_orientations = {}  # Multi-robot orientations
        self.cache_robot_eef_positions = {}  # Multi-robot end-effector positions
        self.cache_robot_base_orientations = {}  # Multi-robot base orientations
        self.cache_robot_hand_orientations = {}  # Multi-robot hand orientations
        self.cache_all_images: Dict[str, np.ndarray] = {}
        self.image_keys_mapping = config.video.image_keys_mapping

    def env_func(self):
        # Create the environment
        env = robosuite.make(self.task_name, **self.robosuite_args)

        # Add visualization wrapper
        env = VisualizationWrapper(env)

        # Add reset wrapper
        env = ResetWrapper(env)

        # Add environment state wrapper
        env = EnvStateWrapper(env)

        # Add success wrapper
        env = SuccessWrapper(env)

        return env

    def initialize(self, *args, **kwargs):
        """Initialize the Robosuite environment."""
        try:
            # Set up robot and task configuration
            self.robot_names = deepcopy(list(self.config.robot.names.values()))
            self.task_name = self.config.task

            # Build robosuite environment arguments
            self.robosuite_args = dict(
                use_camera_obs=False,
                reward_shaping=False,
                has_renderer=self.use_rendering,
                has_offscreen_renderer=self.use_offscreen_rendering,
                control_freq=100,
                ignore_done=True,
                robots=self.robot_names
            )

            # Handle two-arm configurations
            if "Two" in self.task_name:
                self.robosuite_args["env_configuration"] = self.config.env_configuration

            controller_json_path = pjoin(os.path.dirname(os.path.realpath(__file__)), "../assets/osc/robosuite/osc_v1.json")
            with open(controller_json_path, "r") as f:
                controller_args = json.load(f)

            self.robosuite_args["controller_configs"] = controller_args
            self.robosuite_args["control_freq"] = self.config.control.rate

            self.robosuite_args["use_camera_obs"] = True
            camera_names = [key[: -len("_image")] if key.endswith("_image") else key for key in self.image_keys_mapping.keys()]
            self.robosuite_args["camera_names"] = camera_names
            self.robosuite_args["camera_heights"] = [self.config.video.height for _ in camera_names]
            self.robosuite_args["camera_widths"] = [self.config.video.width for _ in camera_names]

            # Handle LIBERO environments
            if "libero" in self.config.task:
                self._setup_libero_environment()

            self.env = StackSubprocVectorEnv([self.env_func for i in range(self.config.num_envs)])

            print(f"Robosuite environment initialized with task: {self.task_name}")
            return True

        except Exception as e:
            print(f"Error initializing Robosuite environment: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def _setup_libero_environment(self):
        """Setup LIBERO environment configuration."""
        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs.bddl_utils import robosuite_parse_problem

            if self.config.robot.task.bddl_tuple is not None:
                # Load from benchmark
                task_suite_name = self.config.robot.task.bddl_tuple[0]
                task_id = self.config.robot.task.bddl_tuple[1]

                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict[task_suite_name]()

                task = task_suite.get_task(task_id)
                self.task_name = task.name
                print(f"LIBERO Task name: {self.task_name}")

                bddl_file_path = os.path.join(
                    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
                )
            else:
                # Load from direct file path
                bddl_file_path = self.config.robot.task.bddl_file

            print(f"LIBERO BDDL file: {bddl_file_path}")

            # Parse the BDDL file to get environment name
            self.task_name = robosuite_parse_problem(bddl_file_path)["problem_name"].title()

            # Add BDDL file to robosuite args
            self.robosuite_args["bddl_file_name"] = bddl_file_path

        except ImportError:
            raise ImportError("LIBERO library not available. Please install libero to use LIBERO tasks.")

    def get_image_data(self):
        return self.cache_all_images

    def normalize_image_views(self, raw_images: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        if len(raw_images) != len(self.image_keys_mapping):
            raise ValueError(
                f"Expected {len(self.image_keys_mapping)} image views, but got {len(raw_images)}. "
                f"Raw keys: {list(raw_images.keys())}, expected keys mapping: {self.image_keys_mapping}"
            )

        normalized = {}
        for key, value in raw_images.items():
            if key in self.image_keys_mapping:
                normalized[self.image_keys_mapping[key]] = value

        # create a list of dicts
        image_data_per_env = []
        for env_id in range(self.config.num_envs):
            image_data_per_env.append({
                view_name: normalized[view_name][env_id] for view_name in self.image_keys_mapping.values()
            })

        return image_data_per_env

    def step_environments(self, actions: np.ndarray) -> Tuple[Any, ...]:
        """Step the environments with the given actions."""
        # Step the environment
        _, reward, done, info = self.env.step(actions)
        updated_obs = self.env.get_env_state()
        return updated_obs, reward, done, None, info

    def get_actions(self, batched_controls: List[Dict[str, Dict[str, Any]]], env_ids: List[int]) -> np.ndarray:
        """
        Vectorized: Get actions for the specified environments based on the batched controls.
        batched_controls: List of control dictionaries for each environment.
        ex: [
            {  // env1
                "device1": {
                    "position": ...,
                    "rotation": ...,
                    "gripper": ...
                    },
                "device2": {
                    "position": ...,
                    "rotation": ...,
                    "gripper": ...
                    }
            },
            {  // env2
                "device3": {
                    "position": ...,
                    "rotation": ...,
                    "gripper": ...
                    },
                "device4": {
                    "position": ...,
                    "rotation": ...,
                    "gripper": ...
                    }
            }
        ]
        """
        num_envs = len(batched_controls)

        positions = []
        rotations = []
        grippers = []
        z_rots = []

        for env_id, env_controls in zip(env_ids, batched_controls):
            for device_idx, device in enumerate(env_controls.keys()):
                positions.append(env_controls[device]["position"])
                rotations.append(env_controls[device]["rotation"])
                grippers.append(env_controls[device]["gripper"])

                if T is not None:
                    q = self.cache_robot_base_orientations[
                        f"robot{device_idx}_base_ori"
                    ][env_id]
                    z_rots.append(T.quat2axisangle(q)[2])
                else:
                    z_rots.append(0.0)

        # ---- Convert to arrays ----
        positions = np.asarray(positions)        # (N, 3)
        rotations = np.asarray(rotations)        # (N, 3)
        grippers = np.asarray(grippers)          # (N, 1)
        z_rots = np.asarray(z_rots)              # (N,)

        # ---- Build rotation matrices ----
        cos = np.cos(z_rots)
        sin = np.sin(z_rots)

        rot_mats = np.zeros((len(z_rots), 3, 3))
        rot_mats[:, 0, 0] = cos
        rot_mats[:, 0, 1] = -sin
        rot_mats[:, 1, 0] = sin
        rot_mats[:, 1, 1] = cos
        rot_mats[:, 2, 2] = 1.0

        # ---- Rotate controls ----
        positions[:, :2] = np.einsum(
            "nij,nj->ni",
            rot_mats[:, :2, :2],
            positions[:, :2],
        )

        rotations = np.einsum(
            "nij,nj->ni",
            rot_mats,
            rotations,
        )

        # ---- Concatenate per-device actions ----
        per_device_actions = np.concatenate([positions, rotations, grippers], axis=1)

        # ---- Concatenate devices per environment ----
        actions = per_device_actions.reshape(num_envs, -1)
        actions = np.clip(actions, -1.0, 1.0)

        # ---- Scatter into full env action array ----
        final_actions = np.zeros(
            (self.config.num_envs, self.config.action_dim),
            dtype=actions.dtype,
        )
        final_actions[env_ids] = actions

        return final_actions

    def reset_environments(self, env_ids=None):
        """Reset the specified environments."""
        if env_ids:
            print("RESETTING ENVS: " + str(env_ids))
        self.env.reset_envs(env_ids)
        return self.env.get_env_state()

    def is_running(self):
        """Check if the simulator is still running."""
        # We assume the simulation never ends unless the user completes their task.
        return True

    def get_environment_state(self, is_relative=True):
        return {"state": self.cache_sim_state}

    def get_robot_state(self):
        """Get current robot state information."""
        self.cache_robot_state.pop('sim_state', None)

        for key in self.image_keys_mapping:
            if key in self.cache_robot_state:
                self.cache_robot_state.pop(key, None)

        # Hacky fix to ensure consistency in observation naming between IsaacLab and Robosuite (only valid for single arm tasks)
        self.cache_robot_state["eef_pos"] = self.cache_robot_state.get("robot0_eef_pos", None)
        self.cache_robot_state["eef_quat"] = self.cache_robot_state.get("robot0_eef_quat", None)

        return self.cache_robot_state

    def get_robot_eef_position(self):
        """Get current end-effector position."""
        return self.cache_robot_eef_positions

    def get_robot_eef_orientation(self):
        """Get current end-effector orientation."""
        return self.cache_robot_eef_orientations

    def get_robot_eef_orientation_by_device_id(self, device_id: int) -> np.ndarray:
        """Get end-effector orientation for a specific robot/device.
        This is in the base frame of the respective robot so that it can be used to properly transform actions

        Args:
            device_id: The device/robot index (0 for first robot, 1 for second, etc.)

        Returns:
            End-effector orientation (quaternion) for the specified device
        """
        return self.cache_robot_hand_orientations[f"robot{device_id}_hand_orn"]

    def update_cache(self, obs):
        """Update cached values"""
        self.cache_sim_state = obs["sim_state"]
        self.cache_robot_state = obs

        raw_image_views = {
            key: value for key, value in obs.items() if key in self.image_keys_mapping
        }
        self.cache_all_images = self.normalize_image_views(raw_image_views)

        # Cache all robot end-effector orientations for multi-robot support
        self.cache_robot_eef_orientations = {}
        self.cache_robot_eef_positions = {}
        self.cache_robot_base_orientations = {}

        for i in range(self.config.num_device):
            pos_key = f"robot{i}_eef_pos"
            quat_key = f"robot{i}_eef_quat"
            base_ori_key = f"robot{i}_base_ori"
            hand_ori_key = f"robot{i}_hand_orn"

            self.cache_robot_eef_orientations[quat_key] = obs[quat_key]
            self.cache_robot_eef_positions[pos_key] = obs[pos_key]
            self.cache_robot_base_orientations[base_ori_key] = obs[base_ori_key]
            self.cache_robot_hand_orientations[hand_ori_key] = obs[hand_ori_key]

    def close(self):
        """Close the simulator."""
        self.env.close()

    def get_metadata(self):
        """
        Used in data collection to record any information needed for postprocessing.
        """
        return dict(
            robosuite_args=self.robosuite_args,
            task_name=self.task_name,
        )

    @property
    def num_environments(self) -> int:
        """Get the number of environments."""
        return self.env.num_envs if self.env else self.config.num_envs

    @property
    def action_dimension(self) -> int:
        """Get the action space dimension."""
        return self.config.action_dim

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self.env.device if self.env else self.config.device
