"""
Simulator factory for creating appropriate simulator instances.
"""

from typing import Any, Type
from .base_simulator import BaseSimulator


class SimulatorFactory:
    """Factory for creating simulator instances."""

    _simulators = {}

    @classmethod
    def register_simulator(cls, name: str, simulator_class: Type[BaseSimulator]) -> None:
        """Register a simulator implementation.

        Args:
            name: Name of the simulator (e.g., 'isaaclab', 'robosuite')
            simulator_class: Simulator class to register
        """
        cls._simulators[name.lower()] = simulator_class

    @classmethod
    def create_simulator(cls, config: Any, *args, **kwargs) -> BaseSimulator:
        """Create a simulator instance.

        Args:
            simulator_type: Type of simulator to create
            config: Configuration object
            *args, **kwargs: Additional arguments for simulator initialization

        Returns:
            Simulator instance

        Raises:
            ValueError: If simulator type is not registered
        """

        simulator_type = config.simulator.name.lower()

        if simulator_type not in cls._simulators:
            available = list(cls._simulators.keys())
            raise ValueError(
                f"Unknown simulator type: {simulator_type}. "
                f"Available simulators: {available}"
            )

        simulator_class = cls._simulators[simulator_type]
        return simulator_class(config, *args, **kwargs)


# Auto-register available simulators
def _register_simulators():
    """Register all available simulator implementations."""
    try:
        from .isaaclab_simulator import IsaacLabSimulator

        SimulatorFactory.register_simulator('isaaclab', IsaacLabSimulator)
    except ImportError as e:
        print(f"Warning: IsaacLab simulator not available (missing dependencies): {e}")

    # Register Robosuite simulator
    try:
        from .robosuite_simulator import RobosuiteSimulator

        SimulatorFactory.register_simulator('robosuite', RobosuiteSimulator)
    except ImportError as e:
        print(f"Warning: Robosuite simulator not available (missing dependencies): {e}")

    # Register YAM Real simulator
    try:
        from .yam_real_simulator import YAMRealSimulator

        SimulatorFactory.register_simulator('YAM_real', YAMRealSimulator)
    except ImportError as e:
        print(f"Warning: YAM simulator not available (missing dependencies): {e}")


# Register simulators on module import
_register_simulators()
