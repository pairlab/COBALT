from typing import Any, Dict, Optional

import utils as U
from controllers.controller import Controller
from data_collection.data_collector import DataCollector

from .device_state import DeviceState
from .video_stream import VideoStream

import time


class Session:
    def __init__(
        self,
        session_id: str,
        config: Dict[str, Any],
        devices: Dict[str, DeviceState],
        data_collector: Optional[DataCollector] = None,
        controller: Optional[Controller] = None,
        env_idx: Optional[int] = None,
        init_image_data: Any = None,
    ) -> None:

        self.session_id = session_id
        self.config = config
        self.devices = devices
        self.data_collector = data_collector
        self.controller = controller
        self.env_idx = env_idx
        self.is_ready = False
        self.is_done = False
        self.num_successes = 0
        self.target = config.robot.task.target
        self.first_trial = (
            True  # Flag used to indicate required env reset before a new client starts
        )

        self.task_complete = False
        self.session_reset_in_progress = False

        self.timer = U.FunctionTimer()
        self.counter = 0

        # For timeout
        self.start_time = None
        self.has_engaged = False

        # For streaming
        if self.config.video.stream:
            self.video_stream = VideoStream(self.config, self.session_id)
            self.video_stream.start()
            self.send_frames(init_image_data)

    def update(self, client_data: Dict[str, Any]) -> None:
        """
        Update session state based on parsed client data.

        Args:
            client_data: Parsed data from DeviceState.
        """

        # Variable to track if any user has engaged control
        user_has_engaged = False

        robot = self.controller.robot

        # Check if any devices have reset flags or task completion or timeout flags pending
        # If so, the session is currently in an invalid state
        for user_index, (device_id, device) in enumerate(self.devices.items()):
            device_client_data = client_data.get(device_id, {})
            device_engaged = bool(device_client_data.get("engaged", False))

            if hasattr(robot, "reset_user_teleop_state") and not device_engaged:
                robot.reset_user_teleop_state(user_index)

            if (
                device.get_doing_reset_handshaking()
                or device.get_task_completion_flag()
                or device.get_task_timeout_flag()
            ):

                self.is_ready = False

                return

            user_has_engaged = user_has_engaged or device_engaged

        # Session is also invalid if no users have engaged
        if not user_has_engaged:
            self.is_ready = False
            return

        self.is_ready = True

    def start_trial(self) -> None:
        """
        Begin a new trial.
        """
        if self.data_collector:
            self.data_collector.start_episode()
        if self.controller:
            self.controller.start_trial()

        self.is_ready = True

        self.set_task_complete(False)
        self.set_session_reset_in_progress(False)

    def end_trial(self, reset, timeout, success, disconnected=False) -> None:
        """
        End the current trial.
        """
        # End data collection and trial
        if self.data_collector:
            self.data_collector.end_episode(success, reset, timeout, disconnected)
        if self.controller:
            self.controller.end_trial()

        self.is_ready = False
        self.stop_timer()

        # Reset the state of all session devices appropriately
        if success:
            self.num_successes += 1
            self.indicate_task_completion_to_clients()
        elif reset:
            self.indicate_reset_to_clients()
        elif timeout:
            self.indicate_timeout_to_clients()
        elif self.get_first_trial():
            self.first_trial = False  # Set False after the initial user connection

        if self.num_successes >= self.target or disconnected:
            self.is_done = True
        else:
            self.is_done = False

    def parse_message(self, msgs) -> Dict[str, Any]:
        """
        Parse a message from the client.

        Returns:
            Dict[str, Any]: Parsed message data
        """
        client_data = {}
        for i, (device_id, device) in enumerate(self.devices.items()):
            client_data[device_id] = device.parse_message(msgs[i])
        return client_data

    def get_session_reset_required(self):
        """
        Get whether the session needs to be reset.
        We use the "session_reset_in_progress" to prevent multiple resets due to the same flag.
        These multiple resets can occur if the client is delayed in sending its reset response acknowledgement.
        """
        if self.session_reset_in_progress:
            return False, (False, False, False)

        first_trial = self.get_first_trial()
        reset = self.get_reset()
        timeout = self.get_timeout()
        task_complete = self.get_task_complete()

        if first_trial or reset or timeout or task_complete:
            self.session_reset_in_progress = True
            return True, (reset, timeout, task_complete)

        return False, (False, False, False)

    def set_session_reset_in_progress(self, status: bool) -> None:
        """
        Set the reset in progress status.

        Args:
            status (bool): True if reset is in progress, False otherwise.
        """
        self.session_reset_in_progress = status

    def get_reset(self):
        """
        Gets whether the session should be reset due to a reset flag from the client.

        Returns:
            bool: True if the session needs to be reset, False otherwise.
        """

        reset_needed = False

        # If any device has a reset flag, the session needs to be reset.
        # We also set the "reset" flag to NO_RESET after ending trial, so that we don't get multiple resets.
        for device in self.devices.values():
            if device.get_reset_flag() != U.ResetFlag.NO_RESET:
                reset_needed = True

        return reset_needed

    def get_timeout(self) -> bool:
        """
        Get whether the session has timed out.

        Returns:
            bool: True if the session has timed out, False otherwise.
        """

        if (
            self.config.robot.task.timeout > 0
            and self.has_engaged
            and self.start_time
            and time.time() - self.start_time > self.config.robot.task.timeout
        ):
            return True

        return False

    def start_timer(self) -> None:
        self.has_engaged = True
        self.start_time = time.time()

    def stop_timer(self) -> None:
        self.has_engaged = False
        self.start_time = None

    def get_task_complete(self) -> bool:
        """
        Get whether the task has been completed.

        Returns:
            bool: True if the task has been completed, False otherwise.
        """
        return self.task_complete

    def indicate_task_completion_to_clients(self) -> None:
        """
        Indicate to clients that the task has been completed.
        """
        for device in self.devices.values():
            device.set_task_completion_flag(True)

    def indicate_reset_to_clients(self) -> None:
        """
        Indicate to clients that a reset is required.
        This function is required because we need to inform ALL OTHER connected clients to reset, too.
        """
        for device in self.devices.values():
            if device.get_reset_flag() == U.ResetFlag.NO_RESET:
                # We tell all other clients that their session is completed so that they reset
                # We can't use the USER_RESET flag because this requires handshaking and assumes the user initiated the reset
                device.set_task_completion_flag(True)
            else:
                device.set_reset_flag(U.ResetFlag.NO_RESET)

    def indicate_timeout_to_clients(self) -> None:
        """
        Indicate to clients that the session has timed out.
        """
        for device in self.devices.values():
            device.set_task_timeout_flag(True)

    def set_task_complete(self, status) -> None:
        """
        Update the task completion status.
        """
        self.task_complete = status

    def get_first_trial(self) -> bool:
        """
        Get whether the current trial is the first trial.

        Returns:
            bool: True if the current trial is the first trial, False otherwise.
        """
        return self.first_trial

    def get_response(self) -> Dict[str, Any]:
        """
        Get the current response data for the session.

        Returns:
            Dict[str, Any]: Response data including task status
        """
        return {
            device_id: self.devices[device_id].get_response()
            for device_id in self.devices
        }

    def get_is_done(self) -> bool:
        """
        Get whether the session is done.

        Returns:
            bool: True if the session is done, False otherwise.
        """
        return self.is_done

    def send_frames(self, frames: dict) -> None:
        """
        Send video frames to the video stream process.

        Args:
            frames (dict): A dictionary of view names to video frames.
        """
        if self.config.video.stream and self.video_stream:
            self.video_stream.update_frame(frames)

    def stop_video_stream(self) -> None:
        """
        Stop the video stream process.
        """
        if self.config.video.stream and self.video_stream:
            self.video_stream.stop()
            del self.video_stream
