"""
Real Robot simulator implementation for the teleoperation server.

Communicates with the CobaltBridgeNode (ROS 2) via ZMQ + msgpack.
The ROS node exposes:
    - A ZMQ REP socket for commands (default tcp://0.0.0.0:5559)

IMPORTANT: The CobaltBridgeNode should be configured with delta_mode='world'
so that position/rotation deltas from the teleop controller (which are in the
robot's base frame after z-rotation correction) are applied correctly.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from copy import deepcopy
import traceback
import time
import cv2
import zmq
from collections.abc import Mapping

from .base_simulator import BaseSimulator

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

try:
    import robosuite.utils.transform_utils as T
except ImportError:
    T = None

try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None


# ── Arm ↔ device mapping ────────────────────────────────────────────────────
DEVICE_TO_ARM = {0: "left", 1: "right"}
ARM_TO_DEVICE = {"left": 0, "right": 1}
ARM_SIDES = ("left", "right")


def _quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion → 3×3 rotation matrix (via scipy)."""
    if R is not None:
        return R.from_quat(q_xyzw).as_matrix().astype(np.float64)
    # Fallback: identity
    return np.eye(3, dtype=np.float64)


class YAMRealSimulator(BaseSimulator):
    """Real YAM Robot hardware interface using ZMQ + msgpack to CobaltBridgeNode."""

    def __init__(self, config: Any, **kwargs):
        super().__init__(config)

        # ── ZMQ addresses (must match CobaltBridgeNode parameters) ───────
        self.zmq_command_address = config.robot.get(
            "zmq_command_address", "tcp://192.168.1.101:5559"
        )
        # ZMQ context and sockets
        self.zmq_context = None
        self.zmq_command_socket = None  # REQ  → ROS node REP

        # Connection state
        self.is_connected = False

        # Robot configuration
        self.task_name = config.task
        self.robot_names = deepcopy(list(config.robot.names.values()))

        assert (len(set(self.robot_names)) == 1), "All robot names must be the same for real robot simulator"

        self.num_robots = len(self.robot_names)

        assert config.num_envs == 1, "Real robot simulator only supports num_envs=1"

        # Control parameters
        self.control_freq = (
            config.control.rate if hasattr(config.control, "rate") else 20
        )
        self.last_command_time = time.time()

        # ── Cached robot state ───────────────────────────────────────────
        self.cache_sim_state = np.zeros(1, dtype=np.float64)  # placeholder sim state
        self.cache_robot_state: Dict[str, Any] = {}
        self.cache_robot_eef_orientations: Dict[str, np.ndarray] = {}
        self.cache_robot_eef_positions: Dict[str, np.ndarray] = {}
        self.cache_robot_base_orientations: Dict[str, np.ndarray] = {}
        self.cache_robot_hand_orientations: Dict[str, np.ndarray] = {}
        self.cache_joint_positions: Dict[str, np.ndarray] = {}
        self.cache_joint_velocities: Dict[str, np.ndarray] = {}
        self.cache_gripper_state: Dict[str, Any] = {}
        self.cache_all_images: Dict[str, np.ndarray] = {}
        self.image_keys_mapping = config.video.image_keys_mapping

    # ── Initialization ───────────────────────────────────────────────────

    def initialize(self, *args, **kwargs):
        """Initialize ZMQ connections to the CobaltBridgeNode."""
        if msgpack is None or msgpack_numpy is None:
            print(
                "ERROR: msgpack and msgpack_numpy are required for "
                "RealRobotSimulator.  pip install msgpack msgpack-numpy"
            )
            return False

        try:
            print("Initializing ZMQ connections to CobaltBridgeNode …")
            print(f"  Command  (REQ → REP): {self.zmq_command_address}")

            self.zmq_context = zmq.Context()

            # Command socket (REQ) → connects to ROS node's REP socket
            self.zmq_command_socket = self.zmq_context.socket(zmq.REQ)
            self.zmq_command_socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.zmq_command_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_command_socket.connect(self.zmq_command_address)

            self.is_connected = True

            # Verify connection with a ping
            ping_response = self._send_command({"cmd": "ping"})
            if ping_response.get("status") != "ok":
                print(f"WARNING: ping to CobaltBridgeNode failed: {ping_response}")
                raise ConnectionError("Failed to ping CobaltBridgeNode")
            else:
                print(f"  Connected - node time: {ping_response.get('node_time_ns')}")

            # Cache static base-frame orientations used in action transforms
            self._refresh_base_orientations()

            print(f"Successfully connected to YAM robot: {self.task_name}")
            return True

        except Exception as e:
            print(f"Error initializing robot connections: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            self.is_connected = False
            return False

    # ── ZMQ helpers ──────────────────────────────────────────────────────

    def _send_command(self, command: dict) -> dict:
        """Send a msgpack-encoded command via ZMQ REQ and receive response."""
        try:
            request_raw = msgpack.packb(command, use_bin_type=True)
            self.zmq_command_socket.send(request_raw)
            response_raw = self.zmq_command_socket.recv()
            return msgpack.unpackb(response_raw, raw=False)
        except zmq.Again:
            return {"status": "error", "error": "Command timeout"}
        except Exception as e:
            print(f"Error sending ZMQ command: {e}")
            return {"status": "error", "error": str(e)}

    # ── Observation processing ───────────────────────────────────────────

    def _process_observation(self, obs: dict):
        """Parse a CobaltBridgeNode observation dict and update all caches."""

        for arm in ARM_SIDES:
            device_id = ARM_TO_DEVICE[arm]

            # ── EEF position / orientation ───────────────────────
            if "eef" in obs and arm in obs["eef"]:
                eef = obs["eef"][arm]
                pos = np.asarray(eef["position"], dtype=np.float64)
                quat_xyzw = np.asarray(eef["quat_xyzw"], dtype=np.float64)

                self.cache_robot_eef_positions[f"robot{device_id}_eef_pos"] = pos[np.newaxis, ...]

                # Quaternion stored as xyzw (robosuite / ROS convention)
                self.cache_robot_eef_orientations[f"robot{device_id}_eef_quat"] = (quat_xyzw[np.newaxis, ...])

                # Hand orientation as (1, 3, 3) rotation matrix so that
                # server's [session.env_idx] indexing returns (3, 3).
                rot_mat = _quat_xyzw_to_rotmat(quat_xyzw)
                self.cache_robot_hand_orientations[f"robot{device_id}_hand_orn"] = (rot_mat[np.newaxis, ...])

            # ── Joint states ─────────────────────────────────────
            if "joints" in obs and arm in obs["joints"]:
                jdata = obs["joints"][arm]
                self.cache_joint_positions[f"robot{device_id}_joint_pos"] = np.asarray(jdata["position"], dtype=np.float64)[np.newaxis, ...]
                self.cache_joint_velocities[f"robot{device_id}_joint_vel"] = np.asarray(jdata["velocity"], dtype=np.float64)[np.newaxis, ...]

        # ── Images ───────────────────────────────────────────────
        if "images" in obs:
            decoded_views = {}
            for key, img_data in obs["images"].items():
                decoded = self._decode_image(img_data)
                if decoded is not None:
                    decoded_views[key] = decoded

            if decoded_views:
                self.cache_all_images = self.normalize_image_views(decoded_views)

        # ── Rebuild combined state dict ──────────────────────────
        self.cache_robot_state = {
            **self.cache_joint_positions,
            **self.cache_joint_velocities,
            **self.cache_robot_eef_positions,
            **self.cache_robot_eef_orientations,
            **self.cache_robot_base_orientations,
            **self.cache_robot_hand_orientations,
            **self.cache_gripper_state,
        }

        # Store as a flat numpy array (same contract as robosuite's sim_state)
        # so that data_collector._create_new_file can index it as state[env_id].
        self.cache_sim_state = np.array(
            [obs.get("timestamp_ns", time.time() * 1e9)], dtype=np.float64
        )

    @staticmethod
    def _decode_image(img_data: dict) -> Optional[np.ndarray]:
        """Decode an image payload from the CobaltBridgeNode."""
        try:
            encoding = img_data.get("encoding", "")
            if encoding == "jpeg":
                buf = (
                    img_data["data"]
                    if isinstance(img_data["data"], bytes)
                    else bytes(img_data["data"])
                )
                img_arr = np.frombuffer(buf, dtype=np.uint8)
                bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif encoding == "raw_bgr8":
                data = img_data["data"]
                shape = img_data.get("shape")
                if isinstance(data, np.ndarray):
                    bgr = data.reshape(tuple(shape)) if shape is not None else data
                else:
                    bgr = np.frombuffer(data, dtype=np.uint8).reshape(tuple(shape))
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Image decode error: {e}")
        return None

    # ── State refresh (on-demand via command socket) ─────────────────────

    def _refresh_base_orientations(self):
        """Fetch and cache base frame quaternions used in action transforms."""
        resp = self._send_command(
            {"cmd": "get_base_frame_quat", "arms": list(ARM_SIDES)}
        )
        if resp.get("status") == "ok" and "base_frame_quat_xyzw" in resp:
            base_quats = resp["base_frame_quat_xyzw"]
            for arm in ARM_SIDES:
                device_id = ARM_TO_DEVICE[arm]
                if arm in base_quats and base_quats[arm] is not None:
                    quat_xyzw = np.asarray(base_quats[arm], dtype=np.float64)
                    # Store as xyzw — T.quat2axisangle (robosuite v1.4)
                    # expects (x, y, z, w) convention
                    self.cache_robot_base_orientations[f"robot{device_id}_base_ori"] = (quat_xyzw[np.newaxis, ...])

    # ── BaseSimulator interface: images ──────────────────────────────────

    def get_image_data(self):
        """Get current camera images for all available views.

        Returns a list of length num_envs (always 1) so that callers can
        index by env_idx: ``get_image_data()[session.env_idx]``.
        """
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
                normalized[self.image_keys_mapping[key]] = np.expand_dims(value, axis=0)  # add env dimension

        # create a list of dicts
        image_data_per_env = []
        for env_id in range(self.config.num_envs):
            image_data_per_env.append({
                view_name: normalized[view_name][env_id] for view_name in self.image_keys_mapping.values()
            })

        return image_data_per_env

    # ── BaseSimulator interface: step ────────────────────────────────────

    def step_environments(self, actions: np.ndarray) -> Tuple[Any, ...]:
        """Compute absolute target poses from deltas and send to CobaltBridgeNode."""
        try:
            command: Dict[str, Any] = {
                "cmd": "delta_command",
                "return_observation": True,
            }

            for device_idx in range(self.num_robots):
                arm = DEVICE_TO_ARM.get(device_idx)
                if arm is None:
                    continue

                action_start_idx = device_idx * 7

                dpos = actions[0, action_start_idx : action_start_idx + 3]
                drotvec = actions[0, action_start_idx + 3 : action_start_idx + 6]
                gripper = actions[0, action_start_idx + 6]

                command[arm] = {
                    "dpos": dpos.tolist(),
                    "drotvec": drotvec.tolist(),
                    "gripper": gripper,
                }

            response = self._send_command(command)

            if response.get("status") == "ok" and "observation" in response:
                self._process_observation(response["observation"])

            reward = np.array([0.0])
            done = np.array([False])
            info = {
                "timestamp": time.time(),
                "response_status": response.get("status"),
            }

            return self.cache_robot_state, reward, done, None, info

        except Exception as e:
            print(f"Error stepping robot: {e}")
            traceback.print_exc()
            return self.cache_robot_state, np.array([0.0]), np.array([False]), None, {"error": str(e)}

    # ── BaseSimulator interface: get_actions ─────────────────────────────

    def get_actions(
        self,
        batched_controls: List[Dict[str, Dict[str, Any]]],
        env_ids: List[int],
    ) -> np.ndarray:
        """Convert batched teleop controls into a flat action array (1 x 14).

        The output format per device is ``[dpos(3), drotvec(3), gripper(1)]``.
        ``step_environments`` later parses this back into per-arm payloads for
        the CobaltBridgeNode ``delta_command``.
        """
        if not batched_controls:
            return np.zeros((1, self.config.action_dim))

        num_envs = len(batched_controls)

        positions = []
        rotations = []
        grippers = []
        z_rots = []

        for env_id, env_controls in zip(env_ids, batched_controls):
            for device_idx, device in enumerate(env_controls.keys()):
                position = env_controls[device]["position"]
                rotation = env_controls[device]["rotation"]
                gripper = env_controls[device]["gripper"]

                positions.append(position)
                rotations.append(rotation)
                grippers.append(gripper)

                if T is not None and device_idx < self.num_robots:
                    base_ori_key = f"robot{device_idx}_base_ori"
                    if base_ori_key in self.cache_robot_base_orientations:
                        q = self.cache_robot_base_orientations[base_ori_key][env_id]
                        z_rots.append(T.quat2axisangle(q)[2])
                    else:
                        z_rots.append(0.0)
                else:
                    z_rots.append(0.0)

        positions = np.asarray(positions)   # (N, 3)
        rotations = np.asarray(rotations)   # (N, 3)
        grippers = np.asarray(grippers)     # (N, 1)
        z_rots = np.asarray(z_rots)         # (N,)

        # Z-rotation matrices (rotate XY into robot base frame)
        cos = np.cos(z_rots)
        sin = np.sin(z_rots)
        rot_mats = np.zeros((len(z_rots), 3, 3))
        rot_mats[:, 0, 0] = cos
        rot_mats[:, 0, 1] = -sin
        rot_mats[:, 1, 0] = sin
        rot_mats[:, 1, 1] = cos
        rot_mats[:, 2, 2] = 1.0

        positions[:, :2] = np.einsum(
            "nij,nj->ni", rot_mats[:, :2, :2], positions[:, :2]
        )
        rotations = np.einsum("nij,nj->ni", rot_mats, rotations)

        per_device_actions = np.concatenate([positions, rotations, grippers], axis=1)

        actions = per_device_actions.reshape(num_envs, -1)
        actions = np.clip(actions, -1.0, 1.0)

        return actions  # shape (num_envs, action_dim) — server indexes actions[env_idx]

    # ── BaseSimulator interface: reset ───────────────────────────────────

    def reset_environments(self, env_ids=None):
        """Reset the robot: clear targets then go home."""
        try:
            print("Resetting robot to home position …")

            resp = self._send_command({"cmd": "reset_targets", "arms": list(ARM_SIDES)})
            if resp.get("status") != "ok":
                print(f"reset_targets failed: {resp}")

            resp = self._send_command({"cmd": "go_home", "arms": list(ARM_SIDES)})
            if resp.get("status") != "ok":
                print(f"go_home failed: {resp}")

            # Wait for motion to complete, then fetch current observation
            time.sleep(2.0)
            resp = self._send_command({"cmd": "get_observation"})
            if resp.get("status") == "ok" and "observation" in resp:
                return resp["observation"]

            return {}

        except Exception as e:
            print(f"Error resetting robot: {e}")
            traceback.print_exc()
            return {}

    # ── BaseSimulator interface: queries ─────────────────────────────────

    def is_running(self):
        """Check if the robot connection is still active."""
        return self.is_connected

    def get_environment_state(self, is_relative=True):
        """Get current environment / world state."""
        return {"state": self.cache_sim_state}

    def get_robot_state(self):
        """Get current robot state information."""
        state = self.cache_robot_state.copy()
        state.pop("sim_state", None)

        # Compatibility aliases for single-arm tasks
        if "robot0_eef_pos" in state:
            state["eef_pos"] = state["robot0_eef_pos"]
        if "robot0_eef_quat" in state:
            state["eef_quat"] = state["robot0_eef_quat"]
        return state

    def get_robot_eef_position(self):
        """Get current end-effector positions for all robots."""
        return self.cache_robot_eef_positions

    def get_robot_eef_orientation(self):
        """Get current end-effector orientations for all robots."""
        return self.cache_robot_eef_orientations

    def get_robot_eef_orientation_by_device_id(self, device_id: int) -> np.ndarray:
        """Get end-effector orientation (rotation matrix) for one device.

        Returns shape ``(1, 3, 3)`` so that the server's
        ``[session.env_idx]`` indexing yields a ``(3, 3)`` matrix suitable
        for ``teleop_to_rotation_control``.
        """
        key = f"robot{device_id}_hand_orn"
        default = np.eye(3, dtype=np.float64)[np.newaxis, ...]  # (1, 3, 3)
        return self.cache_robot_hand_orientations.get(key, default)

    def update_cache(self, obs):
        """Update cached values from latest observation payload."""
        if not obs:
            return
        self._process_observation(obs)

    # ── Cleanup ──────────────────────────────────────────────────────────

    def close(self):
        """Close ZMQ connections."""
        print("Closing robot connections …")
        self.is_connected = False

        if self.zmq_command_socket:
            self.zmq_command_socket.close()
        if self.zmq_context:
            self.zmq_context.term()

        print("Robot connections closed.")

    # ── Metadata ─────────────────────────────────────────────────────────

    def get_metadata(self):
        """Get metadata for data collection."""
        return dict(
            robot_type="YAM",
            communication="ZMQ+msgpack",
            zmq_command_address=self.zmq_command_address,
            task_name=self.task_name,
            control_freq=self.control_freq,
            num_robots=self.num_robots,
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def num_environments(self) -> int:
        return 1

    @property
    def action_dimension(self) -> int:
        return self.config.action_dim

    @property
    def device(self) -> str:
        return "cpu"
