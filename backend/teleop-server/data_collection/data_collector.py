import os
import time
import datetime
import h5py
import uuid
import threading
import logging
import numpy as np
import socket
import queue
import json
from collections import defaultdict

import utils as U


class AsyncFlusher(threading.Thread):
    def __init__(self, flush_function):
        super(AsyncFlusher, self).__init__()
        self._flush_queue = queue.Queue()
        self._flush_function = flush_function
        self._running = threading.Event()
        self._running.set()

    def flush(self, buffer_to_flush):
        """
        NOTE: this function should never get blocked!!
        """
        self._flush_queue.put(buffer_to_flush)

    def run(self):
        """
        The main loop for this thread.
        """
        print(f"{self.__class__.__name__}: AsyncFlusher thread started!")
        while self._running.is_set():
            try:
                buffer_to_flush = self._flush_queue.get(timeout=0.1)
                if buffer_to_flush is None:
                    break

                self._flush_function(buffer_to_flush)

                # signal that the task is done for this buffer
                self._flush_queue.task_done()
            except queue.Empty:
                continue

        print(f"{self.__class__.__name__}: AsyncFlusher thread stopped!")

    def stop(self):
        """
        Called by main thread when it is time to terminate.
        """
        print(f"{self.__class__.__name__}: Closing AsyncFlusher thread...")
        self._running.clear()
        self._flush_queue.put(None)
        self.join()


class DataCollector(object):
    """
    Base data collector for simulation.

    Structure of the HDF5 file:

    Each file follows the naming convention:
        "date_{date}_time_{time}-id-{id}.hdf5"

    The hierarchical structure within the file is as follows:

    - data:
        - demo_0:
            - additional_data:
                - counter: {array of counters}
                - enabled: {array of enabled flags}
                - robot_controls:
                    - action:
                        - gripper: {array of gripper commands}
                        - position: {array of position commands}
                        - rotation: {array of rotation commands}
                - robot_states:
                    - device_id:
                        - position: {array of positions}
                        - rotation: {array of rotations}
                - teleop_commands:
                    - device_id:
                        - gripper: {array of gripper commands}
                        - position: {array of position commands}
                        - rotation: {array of rotation commands}
                - timestamps:
                    - control_loop_time: {array of control loop times}
                    - received: {array of received timestamps}
                    - sent: {array of sent timestamps}
            - states: {array of states}
    """

    @staticmethod
    def to_numpy_if_tensor(x):
        """
        Converts torch/tensor objects to numpy arrays if needed, else returns as is.
        """
        if hasattr(x, 'cpu') and hasattr(x, 'numpy'):
            return x.cpu().numpy()
        return x

    def __init__(
        self,
        config,
        simulator,
        user_id,
        collect_obs=False,
        async_flush=True,
        env_id=None,
    ):
        """
        Args:
            config: config object
            robot (TeleopEnv instance): robot used for teleop
            collect_obs (bool): whether to collect observations or not (sensors), along with the simulator states
            async_flush (bool): whether to start a dedicated thread to flush buffer to disk asynchronously
            env_id (int): used to distinguish different processes, so that writing to the same folder will
                not cause problems
        """

        self.init_time_ref = time.time()

        self.logger = logging.getLogger("TeleoperationServer")

        self.config = config
        self.flush_freq = self.config.data_collection.sim.data_flush_freq
        self._collect_obs = collect_obs
        self.image_keys_mapping = self.config.video.image_keys_mapping

        # for ensuring parallel writes to same folder won't cause issues, we add this identifier and
        # append it in the file name
        self.env_id = env_id
        self.user_id = user_id

        self.directory = self.config.data_collection.directory
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # placeholder for current hdf5 file object
        self.hdf5_file = None
        self.data_group = None
        self.episode_group = None

        # in-memory cache
        self.additional_data = []

        self._do_async_flush = async_flush
        if self._do_async_flush:
            self._async_flusher = AsyncFlusher(self._flush_buffer_to_disk)
            self._async_flusher.start()

        # some metadata to store for this user
        self.total_tasks = 0
        self.success_tasks = 0

        # simulator used to control envs
        self.simulator = simulator

        # record total number of collections attempted
        self.total_steps = 0
        self.total_user_samples_collected = 0

        # assume we will start collection soon after initialization
        self.start_episode()

    def _create_new_file(self, timestamp, additional_data):
        """
        Helper function for creating a new hdf5 file for current demonstration.
        The hdf5 file is named according to the time of creation.
        """
        time_str = datetime.datetime.fromtimestamp(timestamp).strftime(
            "date_%m_%d_%Y_time_%H_%M_%S"
        )
        env_id_str = f"-id-{self.env_id}" if self.env_id is not None else ""
        hdf5_path = os.path.join(self.directory, f"{time_str}{env_id_str}.hdf5")

        self.hdf5_file = h5py.File(hdf5_path, "w")
        self.data_group = self.hdf5_file.create_group("data")
        self.data_group.attrs["timestamp"] = timestamp
        self.data_group.attrs["readable_timestamp"] = time_str

        self.episode_group = self.data_group.create_group("demo_0")
        init_state = self.episode_group.create_group("initial_state")

        for key in additional_data["state_dict"]:
            grp = init_state.create_group(key)
            if isinstance(additional_data['state_dict'][key], dict):
                for subkey, subvalue in additional_data["state_dict"][key].items():
                    subkey_group = grp.create_group(subkey)  # robot group
                    for state_name, state_value in subvalue.items():
                        data = self.to_numpy_if_tensor(state_value[[self.env_id]])
                        subkey_group.create_dataset(state_name, data=data)
            else:
                data = self.to_numpy_if_tensor(additional_data["state_dict"][key][self.env_id])
                grp.create_dataset("state", data=data)

        # store teleop config
        self.episode_group.attrs["teleop_config"] = U.json_dump(self.config.to_dict())

        self.logger.info(f"DataCollector: recording data to {hdf5_path}")

    def start_episode(self):
        """
        Bookkeeping to do at the start of each new episode.
        """

        # timesteps in current episode
        self.episode_steps = 0
        self.has_interaction = False
        self.user_samples_collected = 0
        self._transition_count = 0
        self._chunk_count = 0

    def _on_first_interaction(self, additional_data):
        """
        Bookkeeping for first timestep of episode.
        This function is necessary to make sure that logging only happens after the first
        step call to the simulation, instead of on the reset (people tend to call
        reset more than is necessary in code).
        """

        # sanity check
        assert len(self.additional_data) == 0
        assert self._transition_count == 1
        assert self._chunk_count == 0

        # start clock only when user first starts interaction
        self.episode_start_time = time.time()
        self.logger.info(f"DataCollector: starting episode {self.total_tasks}")

        # make a new hdf5 file for this episode
        self._create_new_file(self.episode_start_time, additional_data)

    def _flush_buffer(self):
        """
        Method to flush internal state to disk.
        """
        if not self.additional_data:
            return

        print("*" * 50)
        print(f"Flushing {len(self.additional_data)} samples to disk...")
        print("*" * 50)
        if not self._do_async_flush:
            self._flush_buffer_to_disk(buffer_to_flush=self.additional_data)
        else:
            self._async_flusher.flush(buffer_to_flush=self.additional_data)
        self.additional_data = []

    def _flush_buffer_to_disk(self, buffer_to_flush):
        """
        Helper function for buffer flushing
        """
        assert buffer_to_flush is not None
        states = {}

        # states structure
        # states {
        #     "articulation": {
        #         "robot_name": {
        #             "robot_state_1": {array of positions},
        #             "robot_state_2": {array of positions},
        #         }
        #     },
        #     "rigid_object": {
        #         "object_name": {
        #             "object_state_1": {array of positions},
        #             "object_state_2": {array of positions},
        #         }
        #     }
        # }

        observations = defaultdict(list)
        episode_images_by_view = {view_name: [] for view_name in self.image_keys_mapping.values()}
        additional_data_dict = defaultdict(list)

        for entry in buffer_to_flush:
            data = entry["controller_info"] if "controller_info" in entry else {}
            state = entry.get("state_dict", None)
            images = data.pop("image_data", None)

            if state is not None:
                # Store robot state
                for key in state:
                    if isinstance(state[key], dict):
                        states.setdefault(key, {})
                        subkeys = state[key].keys()
                        for subkey in subkeys:
                            for subkey_k, val in state[key][subkey].items():
                                states[key].setdefault(subkey, {})
                                if states[key][subkey].get(subkey_k) is None:
                                    states[key][subkey][subkey_k] = []
                                states[key][subkey][subkey_k].append(self.to_numpy_if_tensor(val[self.env_id]))
                    else:
                        states.setdefault(key, [])
                        states[key].append(self.to_numpy_if_tensor(state[key][self.env_id]))

            # Store observations
            for key, value in entry["obs"].items():
                observations[key].append(self.to_numpy_if_tensor(value[self.env_id]))
            if images is not None:
                for view_name, view_image in images.items():
                    episode_images_by_view[view_name].append(self.to_numpy_if_tensor(view_image))

            for key, value in data.items():
                # Check if the value is a dictionary
                if isinstance(value, dict):
                    # Flatten the dictionary and append to the additional_data_dict
                    flattened_dict = U.flatten_nested_dict(value, sep="/")
                    for k, v in flattened_dict:
                        additional_data_dict[f"{key}/{k}"].append(self.to_numpy_if_tensor(v))
                else:
                    additional_data_dict[key].append(self.to_numpy_if_tensor(value))

        # write to hdf5
        chunk_idx = f"{self._chunk_count:06d}"
        if states:
            for key in states:
                if isinstance(states[key], dict):
                    for subkey, subvalue in states[key].items():
                        for subkey_k, subvalue_k in subvalue.items():
                            data_array = np.array(self.to_numpy_if_tensor(subvalue_k))
                            self.episode_group.create_dataset(f"states/{key}/{subkey}/{subkey_k}/{chunk_idx}", data=data_array)
                else:
                    data_array = np.array([self.to_numpy_if_tensor(x) for x in states[key]])
                    self.episode_group.create_dataset(f"states/{key}/{chunk_idx}", data=data_array)

        if observations:
            for key, value in observations.items():
                data_array = np.array([self.to_numpy_if_tensor(x) for x in value])
                self.episode_group.create_dataset(f"obs/{key}/{chunk_idx}", data=data_array)
        if episode_images_by_view:
            for view_name, view_images in episode_images_by_view.items():
                data_array = np.array([self.to_numpy_if_tensor(x) for x in view_images])
                self.episode_group.create_dataset(f"images/{view_name}/{chunk_idx}", data=data_array)
        for key, value in additional_data_dict.items():
            data_array = np.array([self.to_numpy_if_tensor(x) for x in value])
            self.episode_group.create_dataset(f"additional_data/{key}/{chunk_idx}", data=data_array)

        self._chunk_count += 1

    def collect(self, additional_data):
        """
        Collect data from the robot.

        :param additional_data: Log the passed data along with data from robot.
        """
        # increase count of collections attempted
        self.episode_steps += 1
        self.total_steps += 1
        self._transition_count += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            assert self.episode_steps == 1
            self._on_first_interaction(additional_data)
            self.has_interaction = True

        self.additional_data.append(additional_data)

        # counts total number of samples collected where user is enabled and command was sent to robot

        controller_info = additional_data["controller_info"]

        # Check if np.any() is True for any key that starts with "enabled/"
        is_any_enabled = any(np.any(value) for key, value in controller_info.items() if key.startswith("enabled/"))

        if is_any_enabled and "robot_controls" in controller_info:
            self.user_samples_collected += 1

        # ensure that the interaction flag is true after having collected a data point
        self.has_interaction = True

        # flush collected data to disk if necessary
        if self._transition_count % self.flush_freq == 0:
            self._flush_buffer()

    def _gather_metadata_for_episode_end(self, is_done, success=False, reset=False, timeout=False, disconnected=False):
        """
        Helper function to collect metadata to record at end of episode.
        """
        # metadata for this episode
        now = datetime.datetime.now()
        is_done = (
            self.config.robot.task.target != -1
            and self.success_tasks >= self.config.robot.task.target
        ) or (
            self.config.robot.task.samples != -1
            and self.total_user_samples_collected >= self.config.robot.task.samples
        )

        episode_metadata = {
            "type": "demo",
            "user_id": self.user_id,
            "task": self.config.task,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "completed": success,
            "reset": reset,
            "timeout": timeout,
            "disconnected": disconnected,
            "date": now.strftime("%Y-%m-%d"),
            "time": time.time() - self.episode_start_time,
            "robot_metadata": self.simulator.get_metadata(),
            "interface": self.config.client.controllers,
            "clock_time": time.time(),
        }

        session_metadata = {
            "type": "session",
            "user_id": self.config.data_collection.user,
            "num_robots": self.config.num_device,
            "task": self.config.task,
            "time": time.time() - self.init_time_ref,
            "num_targets": self.config.robot.task.target,
            "num_success": self.success_tasks,
            "date": now.strftime("%Y-%m-%d"),
            "interface": self.config.client.controllers,
            "verified": is_done,
            "master": socket.gethostname(),
            "hash": str(uuid.uuid4()) if is_done else None,
        }

        return episode_metadata, session_metadata

    def end_episode(self, success=False, reset=False, timeout=False, disconnected=False):
        """
        Do any cleanup at the end of an episode of interaction.
        Returns a list of JSON strings corresponding to metadata for this
        demonstration.
        """
        json_strings = []
        is_done = False

        if not self.has_interaction:
            # if there was no interaction, don't record anything
            self.start_episode()
            return json_strings, is_done

        # Ensure that the buffer is flushed before ending the episode
        self._flush_buffer()
        if self._do_async_flush:
            self._async_flusher._flush_queue.join()

        # Update session statistics
        self.total_tasks += 1
        if success:
            self.success_tasks += 1
        if success:
            self.total_user_samples_collected += self.user_samples_collected

        # gather metadata for this episode
        episode_metadata, session_metadata = self._gather_metadata_for_episode_end(is_done, success, reset, timeout, disconnected)

        if "pose_info" in episode_metadata:
            pose_info = episode_metadata.pop("pose_info")
            for obj, data in pose_info.items():
                for key, value in data.items():
                    data_array = np.array([self.to_numpy_if_tensor(x) for x in value] if isinstance(value, list) else self.to_numpy_if_tensor(value))
                    self.episode_group.create_dataset(f"pose_info/{obj}/{key}", data=data_array)

        json_strings.extend([
            json_dump(
                episode_metadata,
                exc_handler=lambda e, d: self.logger.error(f"Error in Episode Metadata dump: {e}, Data: {d}"),
            ),
            json_dump(
                session_metadata,
                exc_handler=lambda e, d: self.logger.error(f"Error in Session Metadata dump: {e}, Data: {d}"),
            ),
        ])

        self.episode_group.attrs.update(
            {
                "metadata": json_strings[0],
                "session_metadata": json_strings[1],
                "teleop_config": json_dump(
                    self.config.to_dict(),
                    exc_handler=lambda e, d: self.logger.error(f"Error in Teleop Config dump: {e}, Data: {d}"),
                ),
            }
        )

        # Cleanup
        filename = self.hdf5_file.filename if self.hdf5_file is not None else None
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None
            self.data_group = None
            self.episode_group = None
            if (
                self.config.data_collection.delete_task_failures
                and not success
                and filename
            ):
                self.logger.info(f"Deleting file {filename} due to task failure.")
                os.remove(filename)

        print("*" * 50)
        print(f"NUM SUCCESS: {self.success_tasks}")
        print(f"NUM USER SAMPLE THIS EP: {self.user_samples_collected}")
        print(f"NUM USER SAMPLES COLLECTED: {self.total_user_samples_collected}")
        print("*" * 50)

        # start a new episode
        self.start_episode()

        return json_strings, is_done

    def close(self):
        """
        Close the data collector.
        """
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = self.data_group = self.episode_group = None
        if self._do_async_flush:
            self._async_flusher.stop()


def json_dump(data, exc_handler=None):
    try:
        return json.dumps(data, indent=4, default=lambda o: "<not serializable>")
    except Exception as e:
        if exc_handler is not None:
            exc_handler(e, data)
        else:
            print(f"Error in json dump: {e}")
            print(f"Data: {data}")
        return None
