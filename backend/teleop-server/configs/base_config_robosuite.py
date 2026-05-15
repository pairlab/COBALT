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

ROBOTS_ACTION_DIM = {
    "Panda": 7,
}

config_dir = Path(__file__).parent / "tasks"
with open(config_dir / "robosuite_tasks.json", "r") as f:
    TASKS = json.load(f)


class BaseServerConfigRobosuite(CCRConfig):
    def __init__(self, *args, **kwargs):
        super(BaseServerConfigRobosuite, self).__init__(*args, **kwargs)

        # predictable object placements (don't use while collecting training data)
        self.robot.seeded_object_sampling = False

        self.simulator.name = "robosuite"

        # Redis poll interval
        self.redis.hostname = os.getenv("REDIS_HOST", "localhost")
        self.redis.port = os.getenv("REDIS_PORT", 6379)
        self.redis.db = 0
        self.redis.poll_interval = 0.1

        self.robot.type = "RobosuiteRobot"

        self.robot.names = {
            "left": None,
            "right": None,
            "base": None,
            "torso": None,
        }

        self.task_name = os.getenv("TASK")
        assert (self.task_name in TASKS), f"Invalid task name: {self.task_name}. Valid tasks are: {list(TASKS.keys())}"

        self.task = TASKS[self.task_name]["task"]
        self.num_device = TASKS[self.task_name]["num_devices"]                   # 1 for unimanual tasks, 2 for bimanual tasks

        # Currently only support single arm and bimanual tasks
        assert self.num_device in [1, 2], f"Unsupported num_device={self.num_device}. Only 1 or 2 is currently supported."
        for key in ["left", "right"][: self.num_device]:
            self.robot.names[key] = "Panda"                                      # currently default is set to Panda

        self.validate_action_dim()                                               # checks that all specified robots have the same action dims

        self.action_dim = ROBOTS_ACTION_DIM["Panda"] * self.num_device
        self.device = "cpu"
        self.num_envs = 2                                                        # number of parallel environments

        self.robot.task.target = 100                                             # number of task successes before quitting
        self.robot.task.timeout = TASKS[self.task_name]["timeout"]               # timeout in number of seconds for the task
        self.robot.task.samples = -1                                             # number of user samples to collect before quitting

        self.env_configuration = "single-arm-opposed"                            # used for bimanual tasks, this is the only orientation supported for now

        if "libero" in self.task:
            benchmark = self.task.split('-')[0]
            task_id = int(self.task.split('-')[1])
            self.robot.task.bddl_tuple = (benchmark, task_id)                    # specify LIBERO benchmark and task id within benchmark

        self.controller.mode = "osc"

        # Teleoperation image storage settings
        self.store_image_data = True
        self.image_height = 224
        self.image_width = 224

        # control settings
        self.control.rate = 20                                                   # control rate in Hz for writing joint velocities / torques

        # Video Settings
        self.video.stream = True                                                 # whether to stream video or not
        self.video.buffer_size = 1                                               # number of frames to keep in the buffer for streaming
        self.video.fps = 20                                                      # frames per second for video stream
        self.video.codec = "h264"                                                # codec to use for video encoding
        self.video.width = 480                                                   # width of video stream
        self.video.height = 480                                                  # height of video stream
        self.video.image_keys_mapping = {
            "agentview_image": "agentview",
            "robot0_eye_in_hand_image": "robot0_eye_in_hand",
        }

        if self.num_device == 2:                                                 # add second robot camera view for bimanual task
            self.video.image_keys_mapping["robot1_eye_in_hand_image"] = "robot1_eye_in_hand"

        # logging config
        self.logging.level = logging.INFO                                        # level for printing to terminal
        self.logging.debug = False
        self.logging.record_metrics = False
        self.logging.debug_timing = False

        # data collection config
        self.data_collection.enabled = True
        self.data_collection.user = "default_user"

        # whether to delete demonstrations that are task failures immediately or keep them until postprocessing happens
        self.data_collection.delete_task_failures = True

        # base path for where to store demonstrations
        self.data_collection.base_dir = os.getenv("TELEOP_DATA_DIR", "/tmp/data")

        # sim
        self.data_collection.sim.data_flush_freq = 200                           # how frequently to dump data to disk, in terms of calls to @collect inside main loop

        # fill in the blanks
        self.infer_settings()

    def validate_action_dim(self):
        action_dim = None
        for robot in self.robot.names.values():
            if robot is not None:
                assert robot in ROBOTS_ACTION_DIM, f"Action dimension for robot {robot} not specified in ROBOTS_ACTION_DIM"
                if not action_dim:
                    action_dim = ROBOTS_ACTION_DIM[robot]
                else:
                    assert action_dim == ROBOTS_ACTION_DIM[robot], f"All robots must have the same action dimension. Found {action_dim} and {ROBOTS_ACTION_DIM[robot]}"

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
