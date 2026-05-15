"""
Base simulator interface for teleoperation server.
This defines the contract that all simulator implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Mapping
import numpy as np


class BaseSimulator(ABC):
    """Abstract base class for simulator implementations."""

    def __init__(self, config: Any):
        """Initialize the simulator with configuration.

        Args:
            config: Simulator-specific configuration object
        """
        self.config = config
        self.env = None
        self.env_cfg = None
        self.success_term = None

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the simulator environment and related components."""
        pass

    @abstractmethod
    def reset_environments(self, env_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Reset specific environments or all environments.

        Args:
            env_ids: List of environment IDs to reset. If None, reset all.

        Returns:
            Observation data from the reset environments
        """
        pass

    @abstractmethod
    def get_actions(self, batched_controls: Dict[str, Any], env_ids: List[int]) -> Any:
        """Get actions for the specified environments based on the batched controls.

        Args:
            batched_controls: Dictionary containing control information for each environment
            env_ids: List of environment IDs to get actions for

        Returns:
            Actions to be taken in the specified environments
        """
        pass

    @abstractmethod
    def step_environments(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step the simulation environments with given action data.

        Args:
            action_data: Dictionary containing action information such as:
                - batched_controls: List of control dictionaries
                - env_indices: Environment indices to step
                - num_envs: Total number of environments
                - action_dim: Action space dimension

        Returns:
            Dictionary containing step results with keys:
                - actions: Action tensor that was applied
                - observations: New observations
                - rewards: Reward values
                - terminated: Termination flags
                - truncated: Truncation flags
                - info: Additional information
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the simulation is still running.

        Returns:
            True if simulation is running, False otherwise
        """
        pass

    @abstractmethod
    def get_image_data(self) -> Optional[np.ndarray]:
        """Get current camera/image data from the simulator.

        Returns:
            Image data as {view_name: numpy array}, or None if not available
        """
        pass

    @abstractmethod
    def normalize_image_views(self, raw_images: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """Normalize raw image map to {view_name: image_array}."""
        pass

    @abstractmethod
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state information.

        Returns:
            Dictionary containing robot state data (positions, orientations, etc.)
        """
        pass

    @abstractmethod
    def get_robot_eef_position(self) -> Optional[np.ndarray]:
        """Get current end-effector position."""
        pass

    @abstractmethod
    def get_robot_eef_orientation(self) -> Optional[np.ndarray]:
        """Get current end-effector orientation."""
        pass

    @abstractmethod
    def update_cache(self, obs: Dict[str, Any]) -> None:
        """Update cached observation data."""
        pass

    @abstractmethod
    def get_environment_state(self, is_relative: bool = True) -> Dict[str, Any]:
        """Get current environment state.

        Args:
            is_relative: Whether to return relative state information

        Returns:
            Dictionary containing environment state data
        """
        pass

    @abstractmethod
    def close(self):
        """Close the simulator."""
        pass

    @property
    def robot(self):
        """Get the robot interface."""
        return self._robot

    @robot.setter
    def robot(self, value):
        """Set the robot interface."""
        self._robot = value

    @property
    @abstractmethod
    def num_environments(self) -> int:
        """Get the number of environments in the simulator.

        Returns:
            Number of environments
        """
        pass

    @property
    @abstractmethod
    def action_dimension(self) -> int:
        """Get the action space dimension.

        Returns:
            Action space dimension
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Get the device being used by the simulator.

        Returns:
            Device string (e.g., 'cuda:0', 'cpu')
        """
        pass
