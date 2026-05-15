import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from os.path import join as pjoin
from typing import Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

import configs


def create_argument_parser():
    """Create the argument parser with base arguments."""
    parser = argparse.ArgumentParser(
        description="Teleoperation for Isaac Lab environments."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=2,
        help="Number of environments to simulate."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Lift-Cube-Franka-IK-Rel-v0",
        help="Name of the task.",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="Sensitivity factor."
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=True,
        help="Record videos during training.",
    )

    # Legacy Arguments
    parser.add_argument(
        "--robot",
        default="",
        help="The name of the robot to control."
    )
    parser.add_argument(
        "--user_name",
        default="",
        help="The user name"
    )
    parser.add_argument(
        "--record_data",
        action="store_true",
        help="If provided, do data collection."
    )
    parser.add_argument(
        "--config",
        default="BaseServerConfigRobosuite",
        help="name of config class to use",
    )
    parser.add_argument(
        "--dataset_dir",
        default="",
        help="The directory where we are to save the data from data_collector",
    )

    return parser


import utils as U
from configs.config import make_config
from server_utils.redis_manager import RedisManager
from simulators.simulator_factory import SimulatorFactory

TIMER = U.FunctionTimer()


class TeleoperationServer(object):
    """
    This class corresponds to the main teleoperation server.
    It is responsible for receiving commands from a client and
    controlling the robot accordingly.
    """

    def __init__(self, config):
        self.config = config
        self._setup_logging()

        # Initialize Redis session manager
        self.redis_manager = RedisManager(
            host=self.config.redis.hostname,
            port=self.config.redis.port,
            db=self.config.redis.db,
        )

        # Initialize the available environments
        self.redis_manager.available_environments = list(range(self.config.num_envs))

        # Enforce control loop rate
        self.control_loop_rate = U.Rate(self.config.control.rate)

        # Initialize processing threads
        self.executor = ThreadPoolExecutor()

        # Register a handler to handle SIGINTs (Ctrl-C) so we can clean up the listener.
        signal.signal(signal.SIGINT, self._sigint_handler)

        self.logger.info("TeleopServer: Initialized successfully.")
        try:
            self.main_loop()
        except Exception as e:
            self.logger.error(f"TeleoperationServer: Exception in main loop: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            os._exit(1)

    def _setup_logging(self):
        self.log_dir = pjoin(os.path.dirname(os.path.realpath(__file__)), "./log/")

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        time_str = datetime.now().strftime("%m_%d_%H_%M_%S")
        self.log_file = pjoin(self.log_dir, f"{self.__class__.__name__}_{time_str}.log")

        form = "[%(levelname)s - %(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - ] %(message)s"
        formatter = logging.Formatter(form)

        self.logger = logging.getLogger("TeleoperationServer")

        # Clear any existing handlers to avoid interference with Isaac Sim's logging
        self.logger.handlers.clear()

        # Always create a file handler for our application logs
        file_handler = logging.FileHandler(self.log_file, mode="w")
        file_handler.setLevel(logging.DEBUG if self.config.logging.debug else logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Create console handler for terminal output
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(self.config.logging.level)
        console.setFormatter(formatter)
        self.logger.addHandler(console)

        # Set the logger level
        self.logger.setLevel(logging.DEBUG if self.config.logging.debug else logging.INFO)

        # Prevent propagation to avoid Isaac Sim's handlers from capturing our logs
        self.logger.propagate = False

    def boot(self):
        self.logger.debug("Boot started...")

        try:
            self.simulator = SimulatorFactory.create_simulator(config=self.config, parser=parser)
        except Exception as e:
            self.logger.error(f"TeleoperationServer: Exception in booting simulator: {e}")
            raise

        # Initialize the simulator and check if it succeeded
        if not self.simulator.initialize():
            self.logger.error("TeleoperationServer: Simulator initialization failed")
            raise RuntimeError("Simulator initialization failed")

        self.logger.debug("Boot finished.")

    def main_loop(self):
        """
        The main loop for the controller.
        Wait on new teleopration clients and run control sessions when requested.
        """

        self.logger.debug("TeleopServer: Starting main loop...")

        # Boot the teleoperation system.
        self.boot()
        obs = self.simulator.reset_environments()  # Reset all the environments
        self.simulator.update_cache(obs)  # Initialize cache with new observations -- necessary to prevent errors
        self.logger.info("TeleopServer: Booted the system successfully")

        # Update the available environments every 1 second
        last_env_update_time = time.time()
        while self.simulator.is_running():
            current_time = time.time()
            if (current_time - last_env_update_time) >= self.config.redis.poll_interval:
                self.redis_manager.update_sessions(self.config, self.simulator)
                last_env_update_time = current_time

            # Process control for active sessions
            with TIMER.measure("control_loop"):
                active_sessions_data = self.redis_manager.get_active_sessions_data()
                self.control_loop(active_sessions_data)

                # Update stream
                with TIMER.measure("video_stream"):
                    if self.config.video.stream:
                        self.redis_manager.send_stream_data(self.simulator.get_image_data())

            self.control_loop_rate.sleep()

        # End of main loop
        self.logger.info("Main Loop: Terminating...")
        self._on_termination(end_trial=False)

    def update_session_after_reset(
        self,
        session_id: str,
        session_reset_type: tuple,
        pipeline: Any,
    ) -> None:
        session = self.redis_manager.get_session(session_id)

        # Session is not in local cache skip
        if not session:
            return

        # End the trials
        session.end_trial(*session_reset_type)

        # Start a new trial
        session.start_trial()

        self.redis_manager.send_response(session_id, pipeline)

    def update_session_after_step(
        self,
        session_id: str,
        terminated: bool,
        additional_info: Dict[str, Any],
        pipeline: Any,
    ) -> None:

        session = self.redis_manager.get_session(session_id)

        # Session is not in local cache skip
        if not session:
            return

        if not session.has_engaged:
            session.start_timer()

        # Data collection (if enabled)
        if session.data_collector:
            additional_info["controller_info"]["success"] = terminated
            session.data_collector.collect(additional_data=additional_info)

        session.set_task_complete(terminated)

        self.redis_manager.send_response(session_id, pipeline)

    def step_envs(self, batched_controls, to_step_env_idxs):
        """
        Step the environment with the given controls.
        """

        actions = self.simulator.get_actions(batched_controls, to_step_env_idxs)

        # Control the robot
        with TIMER.measure("step_envs"):
            obs, reward, terminated, truncated, info = self.simulator.step_environments(actions)

        return (actions, obs, reward, terminated, truncated, info)

    def process_session_for_step(self, session_id, client_data):
        """
        Process a single session to generate controls and collect additional data.
        Returns None if session is invalid, else (control_dict, session_id, env_idx, client_data).
        """
        session = self.redis_manager.get_session(session_id)
        if not session:
            return None

        if len(client_data) != self.config.num_device:
            self.logger.warning(
                "Skipping session %s due to device mismatch: got %d, expected %d",
                session_id,
                len(client_data),
                self.config.num_device,
            )
            return None

        control_dict = {}
        for idx, (device_id, device_data) in enumerate(client_data.items()):
            # Get robot orientation for this specific device
            robot_orientation = self.simulator.get_robot_eef_orientation_by_device_id(idx)[session.env_idx]

            controls = session.controller.preprocess_user_info(
                user_index=idx,
                user_info=device_data,
                device_id=device_id,
                current_rotation=robot_orientation,
            )
            control_dict[device_id] = controls

        return control_dict, session_id, session.env_idx, client_data

    def control_loop(self, active_sessions_data: Dict[str, Dict]) -> None:
        """
        This function is the main control loop for the robot.
        """
        self.logger.debug("TeleoperationServer: entered control_loop")

        # No active sessions, return
        if not active_sessions_data:
            return

        # Collect sessions for reset and step
        with TIMER.measure("collect_sessions"):
            session_env_idxs_to_reset = []
            to_reset_env_idxs = []
            session_reset_types = []
            sessions_to_step = []
            for session_id, client_data in active_sessions_data.items():
                session = self.redis_manager.get_session(session_id)
                if not session:
                    continue
                session_reset_required, reset_type = (session.get_session_reset_required())

                if session_reset_required:
                    session_env_idxs_to_reset.append((session_id, session.env_idx))
                    to_reset_env_idxs.append(session.env_idx)
                    session_reset_types.append(reset_type)
                elif session.is_ready:
                    sessions_to_step.append((session_id, client_data))

        futures = [
            self.executor.submit(self.process_session_for_step, *args)
            for args in sessions_to_step
        ]
        with TIMER.measure("preprocess_sessions"):
            batched_controls = []
            to_step_env_idxs = []
            batched_session_env_idxs = []
            session_client_data = {}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                control_dict, session_id, env_idx, client_data = result
                batched_controls.append(control_dict)
                to_step_env_idxs.append(env_idx)
                batched_session_env_idxs.append((session_id, env_idx))
                session_client_data[(session_id, env_idx)] = client_data

        # Perform batched resets
        with TIMER.measure("reset_envs"):
            if session_env_idxs_to_reset:
                obs = self.simulator.reset_environments(env_ids=to_reset_env_idxs)
                self.simulator.update_cache(obs)

        # Perform batched stepping (control)
        with TIMER.measure("step_envs_outer"):
            if batched_controls:
                # Retrieve state, image, and observation data before stepping
                state_dict = self.simulator.get_environment_state(is_relative=True)
                image_data = self.simulator.get_image_data()
                obs_data = self.simulator.get_robot_state()

                additional_data = dict()

                ((actions, obs, reward, terminated, truncated, info)) = self.step_envs(
                    batched_controls, to_step_env_idxs
                )

                # Gather additional data from controller and environment (once per session)
                for session_id, env_idx in batched_session_env_idxs:
                    session = self.redis_manager.get_session(session_id)
                    if self.config.data_collection.enabled:
                        additional_data[env_idx] = session.controller.collect_session(
                            user_infos_by_device=session_client_data[(session_id, env_idx)],
                            image_data=image_data[env_idx],
                        )
                        additional_data[env_idx]["state_dict"] = state_dict
                        additional_data[env_idx]["obs"] = obs_data
                        additional_data[env_idx]["controller_info"]["robot_controls"]["actions"] = actions[env_idx]

                # Update state caches AFTER stepping
                self.simulator.update_cache(obs)

        # Pipeline for efficient updatess
        pipeline = self.redis_manager.get_pipeline()

        with TIMER.measure("reset redis update"):
            # Update the session after reset
            if session_env_idxs_to_reset:
                reset_tasks = [
                    (
                        session_id,
                        reset_type,
                        pipeline,
                    )
                    for (session_id, _), reset_type in zip(
                        session_env_idxs_to_reset, session_reset_types
                    )
                ]
                list(
                    self.executor.map(
                        lambda args: self.update_session_after_reset(*args),
                        reset_tasks,
                    )
                )

        with TIMER.measure("step redis update"):
            # Update the session after step
            if batched_controls:
                # Send responses back to clients via Redis & write data to file
                step_tasks = [
                    (
                        session_id,
                        terminated[env_idx].item(),
                        (
                            additional_data[env_idx]
                            if self.config.data_collection.enabled
                            else None
                        ),
                        pipeline,
                    )
                    for (session_id, env_idx) in batched_session_env_idxs
                ]

                list(
                    self.executor.map(
                        lambda args: self.update_session_after_step(*args), step_tasks
                    )
                )

        # Execute the pipeline
        with TIMER.measure("redis pipline"):
            pipeline.execute()

    def _sigint_handler(self, signal, frame):
        """
        Handler for SIGINT. Note that this function is processed by the same thread,
        since it interrupts execution.
        """
        self.logger.debug("TeleoperationServer: got SIGINT")

        try:
            self.simulator.close()
        except Exception as e:
            self.logger.error(f"Error closing application: {e}")

        exit(0)


if __name__ == "__main__":

    # Create the server first so it can boot the simulator and get parsed args

    # For now, create a minimal config to start the server
    parser = create_argument_parser()

    # Parse basic args to get config type
    known_args, unknown_args = parser.parse_known_args()

    # Get overriden simulator information
    sim = os.getenv("SIMULATOR")
    if sim == "robosuite":
        known_args.config = "BaseServerConfigRobosuite"
    elif sim == "isaaclab":
        known_args.config = "BaseServerConfigIsaac"
    elif sim == "YAM_real":
        known_args.config = "BaseServerConfigYAM"
    else:
        print("Invalid simulator specified")
        sys.exit(1)

    # If IsaacLab is selected, fail with a clear optional-dependency error.
    if known_args.config == "BaseServerConfigIsaac":
        isaac_import_error = configs.get_isaac_import_error()
        if isaac_import_error is not None:
            raise RuntimeError(
                "IsaacLab simulator was selected, but IsaacLab dependencies are not installed. "
                "Install IsaacLab dependencies and run the IsaacLab setup path."
            ) from isaac_import_error

    # Create the configuration
    if known_args.config.endswith(".json"):
        # This is a config file, load it
        ext_cfg = json.load(open(known_args.config, "r"))
        print("loading external config: =================")
        print(json.dumps(ext_cfg, indent=4))
        print("==========================================")
        config_class = getattr(configs, ext_cfg["config_class"], None)
        if config_class is None:
            isaac_import_error = configs.get_isaac_import_error()
            if ext_cfg["config_class"] == "BaseServerConfigIsaac" and isaac_import_error is not None:
                raise RuntimeError(
                    "Config requests BaseServerConfigIsaac, but IsaacLab dependencies are not installed."
                ) from isaac_import_error
            raise ValueError(f"Unknown config class: {ext_cfg['config_class']}")
        main_config = config_class()
        main_config.safe_update(ext_cfg)
    else:
        main_config = make_config(known_args.config)

    # Apply basic overrides from known args
    if len(known_args.robot):
        main_config.robot.type = known_args.robot
    if len(known_args.user_name):
        main_config.data_collection.user = known_args.user_name
    if len(known_args.dataset_dir):
        main_config.data_collection.base_dir = known_args.dataset_dir
    if known_args.record_data:
        main_config.data_collection.enabled = known_args.record_data

    main_config.infer_settings()

    # create the server
    server = TeleoperationServer(main_config)
