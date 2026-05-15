"""
Client Configuration for the current teleoperation session.
"""

from configs.config import CCRConfig


class BaseClientConfig(CCRConfig):
    def __init__(self, *args, **kwargs):
        super(BaseClientConfig, self).__init__(*args, **kwargs)

        # client interface and controller id
        self.client.controllers = [("keyboard", 0)]

        self.client.first_person = False   # orient user behind robot instead of in front

        self.client.sample_interval = 0.05 # interval to sample the client at

        # websocket configs
        self.sim.sim_type = "bimanual"
        self.sim.sim = "Robosuite"         # simulator name
        self.sim.arm = "left"              # arm configuration
        self.username = "default"
        self.session_id = None
        self.device_id = None

        # keyboard interface configs
        self.keyboard.delta_pos = 0.01     # how much to change position by on every registered key press
        self.keyboard.delta_roll = 1.0     # how much to change roll by on every registered key press
        self.keyboard.delta_pitch = 0.5    # how much to change pitch by on every registered key press
        self.keyboard.base_ang_vel = 90
        self.keyboard.base_lin_vel = 1.0

        # 3d mouse interface configs
        self.mouse.pos_sensitivity = 0.02  # sensitivity for position movement
        self.mouse.orn_sensitivity = 0.01  # sensitivity for orientation movement

        self.infer_settings()

    def infer_settings(self):
        """
        Should be called any time settings change.
        Computes settings derived from other settings.
        """

        # indicators for client control interfaces
        self.client.flag.keyboard = self.client.controllers[0][0] == "keyboard"
        self.client.flag.mouse = self.client.controllers[0][0] in ["3d_mouse"]
        assert (self.client.flag.keyboard or self.client.flag.mouse), "invalid client setting"
