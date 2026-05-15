"""
Configuration for the current teleoperation session.
"""

import logging
from configs.config import CCRConfig
import os
import json
from pathlib import Path

# dictionary of valid controller schemes and their associated controllers
CONTROLLERS = {
    "osc": "OSCController",
}

config_dir = Path(__file__).parent / "tasks"
with open(config_dir / "robosuite_tasks.json", "r") as f:
    TASKS = json.load(f)


class BaseServerConfigYAM(CCRConfig):
    def __init__(self, *args, **kwargs):
        super(BaseServerConfigYAM, self).__init__(*args, **kwargs)

        # predictable object placements (don't use while collecting training data)
        self.robot.seeded_object_sampling = False

        self.simulator.name = "YAM_real"

        # Redis poll interval
        self.redis.hostname = os.getenv("REDIS_HOST", "localhost")
        self.redis.port = os.getenv("REDIS_PORT", 6379)
        self.redis.db = 0
        self.redis.poll_interval = 0.1

        self.robot.type = "YAMRobot"

        self.robot.names = {
            "left": "YAM",
            "right": "YAM",
            "base": None,
            "torso": None,
        }

        self.task_name = os.getenv("TASK")

        assert (
            self.task_name in TASKS
        ), f"Invalid task name: {self.task_name}. Valid tasks are: {list(TASKS.keys())}"

        self.task = TASKS[self.task_name]["task"]
        self.action_dim = 14
        self.num_device = 2  # 1 for unimanual tasks, 2 for bimanual tasks
        self.device = "cpu"
        self.num_envs = 1

        self.robot.task.target = 100  # number of task successes before quitting
        self.robot.task.timeout = TASKS[self.task_name][
            "timeout"
        ]  # timeout in number of seconds for the task
        self.robot.task.samples = (
            -1
        )  # number of user samples to collect before quitting

        # self.env_configuration = "single-arm-opposed"  # used for bimanual tasks
        self.controller.mode = "osc"

        # Teleoperation image storage settings
        self.store_image_data = True
        self.image_height = 224
        self.image_width = 224

        # control settings
        self.control.rate = (
            20  # control rate in Hz for writing joint velocities / torques
        )

        # Video Settings
        self.video.stream = True  # whether to stream video or not
        self.video.buffer_size = (
            1  # number of frames to keep in the buffer for streaming
        )
        self.video.fps = 20  # frames per second for video stream
        # self.video.codec = "h264_nvenc"  # codec to use for video encoding
        self.video.codec = "h264"  # codec to use for video encoding
        self.video.width = 480  # width of video stream
        self.video.height = 480  # height of video stream
        self.video.image_keys_mapping = {
            "front": "front",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
        }

        # logging config
        self.logging.level = logging.INFO  # level for printing to terminal
        self.logging.debug = False
        self.logging.record_metrics = False
        self.logging.debug_timing = False  # True

        # data collection config
        self.data_collection.enabled = True
        self.data_collection.user = "default_user"

        # whether to delete demonstrations that are task failures immediately or keep them until postprocessing happens
        self.data_collection.delete_task_failures = False

        # base path for where to store demonstrations
        self.data_collection.base_dir = os.getenv("TELEOP_DATA_DIR", "/tmp/data")

        #   sim
        self.data_collection.sim.data_flush_freq = 200  # how frequently to dump data to disk, in terms of calls to @collect inside main loop

        # fill in the blanks
        self.infer_settings()

    def infer_settings(self):
        """
        Should be called any time settings change.
        Computes settings derived from other settings.
        """

        # Filter out unused robot components and filter ports and controllers based on used robot components
        self.robot.names = {key: val for (key, val) in self.robot.names.items() if val}

        # controller class to use
        self.controller.type = CONTROLLERS[self.controller.mode]

        # data directory for storing demonstrations
        self.data_collection.directory = self.data_collection.base_dir

        # indicators for control mode
        self.controller.flag.osc = self.controller.mode == "osc"
        self.controller.flag.joint_velocity = self.controller.mode == "velocity"
        self.controller.flag.joint_torque = self.controller.mode == "torque"
        self.controller.flag.opspace = self.controller.mode == "opspace"
        assert (
            self.controller.flag.osc
            or self.controller.flag.joint_velocity
            or self.controller.flag.joint_torque
            or self.controller.flag.opspace
        ), "invalid control setting"
