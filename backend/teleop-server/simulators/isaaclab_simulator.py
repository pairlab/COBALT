"""
IsaacLab simulator implementation for the teleoperation server.
Contains all IsaacLab-specific logic extracted from the main server.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
import traceback
from collections.abc import Mapping

from .base_simulator import BaseSimulator


class IsaacLabSimulator(BaseSimulator):
    """IsaacLab simulator implementation."""
    
    def __init__(self, config: Any, parser: Any):
        """Initialize IsaacLab simulator.
        
        Args:
            config: Teleop server configuration
            parser: Command line argument parser
        """
        super().__init__(config)
        self.simulation_app = None
        self.cache_robot_state = None
        self.cache_robot_eef_orientations = {}  # Multi-robot orientations
        self.cache_robot_eef_positions = {}  # Multi-robot end-effector positions
        self.cache_robot_base_orientations = {}  # Multi-robot base orientations
        self.cache_robot_hand_orientations = {}  # Multi-robot hand orientations
        self.cache_all_images: Dict[str, np.ndarray] = {}
        self.image_keys_mapping = config.video.image_keys_mapping
        self.args = None  # Will be set after parsing
        
        # Launch Isaac Sim application first - this initializes Isaac dependencies
        from isaaclab.app import AppLauncher
        
        # Add AppLauncher arguments first, then parse
        AppLauncher.add_app_launcher_args(parser)
        self.args = parser.parse_args()

        if self.args.video:
            self.args.enable_cameras = True
            
        app_launcher = AppLauncher(self.args)
        self.simulation_app = app_launcher.app
        
        # Now we can import Isaac-specific modules
        global gym, omni, parse_env_cfg
        import gymnasium as gym
        import omni.log as omni
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    def initialize(self):
        """Initialize the IsaacLab environment."""
        try:
            # Set up robot and task configuration
            self.env_cfg = parse_env_cfg(
                self.config.task, 
                device=self.config.device, 
                num_envs=self.config.num_envs
            )
            
            # Configure environment settings
            self.env_cfg.episode_length_s = 600  # TODO: Move to config
            self.env_cfg.num_envs = self.config.num_envs
            self.env_cfg.observations.policy.concatenate_terms = False

            # Get success checking termination function
            self._setup_success_termination()

            # Make environment run till completion
            if hasattr(self.env_cfg.terminations, "timeout"):
                self.env_cfg.terminations.timeout = None
            if hasattr(self.env_cfg.terminations, "time_out"):
                self.env_cfg.terminations.time_out = None
            
            # Viewer configuration (for non-headless mode)
            self.env_cfg.viewer.eye = (1.8, 0.0, 1.0)
            self.env_cfg.viewer.look_at = (0.0, 0.0, 0.0)
            
            # Setup Gym Environment
            self.env = gym.make(
                self.config.task, 
                cfg=self.env_cfg, 
                render_mode="rgb_array"
            ).unwrapped

            # Update config with environment info
            self.config["env"] = self.env
            self.config["env_cfg"] = self.env_cfg
            self.config["task_name"] = self.config.task
            self.config["success_term"] = self.success_term

            print(f"IsaacLab environment initialized with task: {self.config.task_name}")
            return True
        
        except Exception as e:
            print(f"Error initializing IsaacLab environment: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def _setup_success_termination(self) -> None:
        """Setup success termination function."""
        self.success_term = None
        self.config["success_term"] = None
        
        if hasattr(self.env_cfg.terminations, "success"):
            self.success_term = self.env_cfg.terminations.success
            self.config["success_term"] = self.success_term
        else:
            omni.warn(
                "No success termination function found in the environment configuration."
            )
            raise Exception(
                "No success termination function found in the environment configuration."
            )
    
    def reset_environments(self, env_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Reset specific environments or all environments."""
        with torch.inference_mode():
            if env_ids:
                env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
                obs, _ = self.env.reset(env_ids=env_ids_tensor)
                self.env.sim.render()
            else:
                obs, _ = self.env.reset()
            
            return obs
    
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
            image_data_per_env.append(
                {
                    view_name: normalized[view_name][env_id] for view_name in self.image_keys_mapping.values()
                }
            )

        return image_data_per_env

    def step_environments(self, actions: np.ndarray) -> Tuple[Any, ...]:
        """Step the environments with the given actions."""
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = self.env.step(torch.tensor(actions, device=self.env.device))
            return obs, reward, terminated.cpu().numpy(), truncated, info

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
        per_device_actions = np.concatenate(
            [positions, rotations, grippers], axis=1
        )

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

    def get_robot_state(self) -> Optional[Dict[str, Any]]:
        """Get current robot state information."""
        if self.cache_robot_state is None:
            return None

        state: Dict[str, Any] = {}
        for k, v in self.cache_robot_state.items():
            if torch.is_tensor(v):
                state[k] = v.detach().cpu().numpy()
            else:
                state[k] = v

        # Compatibility aliases for single-arm tasks.
        if "robot0_eef_pos" in state:
            state["eef_pos"] = state["robot0_eef_pos"]
        if "robot0_eef_quat" in state:
            state["eef_quat"] = state["robot0_eef_quat"]

        return state

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
        key = f"robot{device_id}_hand_orn"
        default = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)[np.newaxis, :],
            (self.config.num_envs, 1),
        )  # (num_envs, 4) wxyz
        return self.cache_robot_hand_orientations.get(key, default)

    @staticmethod
    def _quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
        """Quaternion conjugate for wxyz quaternions. Expects (..., 4)."""
        return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)

    @staticmethod
    def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiply for wxyz quaternions. Expects (..., 4)."""
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    def get_environment_state(self, is_relative: bool = True) -> Dict[str, Any]:
        """Get current environment state."""
        return self.env.scene.get_state(is_relative=is_relative)
    
    def is_running(self) -> bool:
        """Check if the simulator is still running."""
        return self.simulation_app.is_running()

    def shutdown(self) -> None:
        """Shutdown the simulation and clean up resources."""
        self.close()

    def update_cache(self, obs: Dict[str, Any]) -> None:
        """Update cached robot state and image data."""
        self.cache_robot_state = obs["policy"]

        # Cache all robot end-effector orientations for multi-robot support
        self.cache_robot_eef_orientations = {}
        self.cache_robot_eef_positions = {}
        self.cache_robot_base_orientations = {}
        self.cache_robot_hand_orientations = {}

        for i in range(self.config.num_device):
            pos_key = f"robot{i}_eef_pos"
            quat_key = f"robot{i}_eef_quat"
            base_ori_key = f"robot{i}_base_ori"
            hand_ori_key = f"robot{i}_hand_orn"
            robot_key = f"robot{i}"
            ee_frame_key = f"robot{i}_ee_frame"
            
            if pos_key in self.cache_robot_state:
                self.cache_robot_eef_positions[pos_key] = self.cache_robot_state[pos_key].cpu().numpy()
            if quat_key in self.cache_robot_state:
                self.cache_robot_eef_orientations[quat_key] = self.cache_robot_state[quat_key].cpu().numpy()
            if base_ori_key in self.cache_robot_state:
                self.cache_robot_base_orientations[base_ori_key] = self.cache_robot_state[base_ori_key].cpu().numpy()
            if hand_ori_key in self.cache_robot_state:
                self.cache_robot_hand_orientations[hand_ori_key] = self.cache_robot_state[hand_ori_key].cpu().numpy()

            # Derive base and end-effector orientations (wxyz) from the scene.
            # Mirrors Robosuite's wrapper-injected `robot{i}_base_ori` and `robot{i}_hand_orn` keys.
            try:
                if base_ori_key in self.cache_robot_base_orientations and hand_ori_key in self.cache_robot_hand_orientations:
                    continue

                if robot_key in self.env.scene.keys() and ee_frame_key in self.env.scene.keys():
                    robot = self.env.scene[robot_key]
                    ee_frame = self.env.scene[ee_frame_key]
                else:
                    robot = self.env.scene["robot"]
                    ee_frame = self.env.scene["ee_frame"]

                base_quat_wxyz = robot.data.root_state_w[:, 3:7]
                ee_quat_wxyz = ee_frame.data.target_quat_w[:, 0, :]

                hand_quat_wxyz = self._quat_mul_wxyz(self._quat_conjugate_wxyz(base_quat_wxyz), ee_quat_wxyz)

                self.cache_robot_base_orientations[base_ori_key] = base_quat_wxyz.detach().cpu().numpy()
                self.cache_robot_hand_orientations[hand_ori_key] = hand_quat_wxyz.detach().cpu().numpy()
            except Exception:
                # If a task doesn't expose these scene entities, teleop still works but skips yaw/hand orientation.
                pass
        
        # Update image cache if video streaming is enabled
        if self.config.video.stream:
            raw_image_views = {}
            for raw_key in self.image_keys_mapping.keys():
                if raw_key in self.env.scene.sensors:
                    raw_image_views[raw_key] = (
                        self.env.scene.sensors[raw_key]
                        .data.output["rgb"]
                        .cpu()
                        .numpy()
                    )
            self.cache_all_images = self.normalize_image_views(raw_image_views)

    def close(self):
        """Close the simulator."""
        self.simulation_app.close()

    def get_metadata(self):
        """
        Used in data collection to record any information needed for postprocessing.
        """
        # Store meta data
        self.metadata = {
            "task_name": self.config.task_name,
        }
        self.metadata.update(self.env.metadata)
        robot_cfgs = {}
        for scene_key in ("robot", "robot0", "robot1"):
            if hasattr(self.env_cfg.scene, scene_key):
                robot_cfgs[scene_key] = getattr(self.env_cfg.scene, scene_key).to_dict()

        if "robot" in robot_cfgs and len(robot_cfgs) == 1:
            self.metadata.update(**robot_cfgs["robot"])
        elif robot_cfgs:
            self.metadata["robots"] = robot_cfgs
        return self.metadata

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
