from typing import Any, Dict
import json

import numpy as np

import utils as U

USER_RESET = U.ResetFlag.USER_RESET


class DeviceState:
    def __init__(self, device_id: str, config: Dict[str, Any]) -> None:
        self.device_id = device_id
        self.config = config

        # Flags that are used by Session to determine the state of the session
        self.reset_flag = False
        self.task_completion_flag = False
        self.task_timeout_flag = False

        # Reset state tracking
        self.last_reset_state = False
        self.reset_indicator = 0  # this indicator is toggled whenever a reset happens
        self.doing_reset_handshaking = (
            False  # if True, disable control until device acknowledges reset
        )

        # Task completion tracking
        self.task_completion_indicator = (
            0  # this indicator is toggled whenever a task is completed
        )

        # Task timeout tracking
        self.task_timeout_indicator = (
            0  # this indicator is toggled whenever a task times out
        )

    def get_response(self):
        return {
            "reset": int(self.reset_indicator),
            "complete": int(self.task_completion_flag),
            "timeout": int(self.task_timeout_flag),
        }

    def parse_message(self, msg):
        """
        Parse a message from a client.
        Returns a dictionary.
        """
        device_msg = json.loads(msg)

        # reset the trial on command (If Client initiates reset)
        reset_state = bool(device_msg.get("reset", 0))
        if (not self.last_reset_state) and reset_state:
            self.reset_flag = USER_RESET
            self.reset_indicator = (self.reset_indicator + 1) % 2
            self.doing_reset_handshaking = True
        # detect the end of the handshaking period
        elif self.last_reset_state and (not reset_state):
            # we should only start the next episode after handshaking is complete
            # and the client state has been reset appropriately
            self.doing_reset_handshaking = False
        self.last_reset_state = reset_state

        # handle task completion acknowledgment (this happens after the server indicates completion)
        task_completion_indicator = device_msg.get("completion", 0)
        if task_completion_indicator != self.task_completion_indicator:
            # when indicator is toggled, task completion has been handled by client,
            # so we can drive the signal low again, and wait to start a new trial
            self.task_completion_flag = False
        self.task_completion_indicator = task_completion_indicator

        # handle task timeout acknowledgment (this happens after the server indicates timeout)
        task_timeout_indicator = device_msg.get("timeout", 0)
        if task_timeout_indicator != self.task_timeout_indicator:
            # when indicator is toggled, task timeout has been handled by client,
            # so we can drive the signal low again, and wait to start a new trial
            self.task_timeout_flag = False
        self.task_timeout_indicator = task_timeout_indicator

        # if user control is enabled
        engaged = bool(device_msg["enable"]) and (not self.doing_reset_handshaking)

        user_displacement = {}

        # message id
        user_displacement["id"] = device_msg["id"]

        # message timestamps
        user_displacement["sent"] = float(device_msg["timestamp"])
        user_displacement["received"] = float(device_msg["received"])
        # change in user position
        user_displacement["dpos"] = device_msg["dpos"]

        # rotation matrix corresponding to target pose
        user_displacement["rotation"] = np.array(device_msg["rotation"]).reshape((3, 3))

        user_displacement["grasp"] = bool(device_msg["grasp"])
        user_displacement["engaged"] = engaged
        user_displacement["valid"] = bool(device_msg["valid"])

        # ADDED: account for phone bug on first few timestamps where engaged and valid can both be True, but
        #        dpos and rotation haven't been updated yet
        if (
            user_displacement["engaged"]
            and user_displacement["valid"]
            and (np.sum(user_displacement["dpos"]) == 0.0)
            and (np.sum(user_displacement["rotation"]) == 3.0)
        ):
            user_displacement["valid"] = False

        return user_displacement

    def get_task_completion_flag(self):
        return self.task_completion_flag

    def get_task_timeout_flag(self):
        return self.task_timeout_flag

    def get_reset_flag(self):
        return self.reset_flag

    def get_doing_reset_handshaking(self):
        return self.doing_reset_handshaking

    def set_task_completion_flag(self, status: bool):
        self.task_completion_flag = status

    def set_task_timeout_flag(self, status: bool):
        self.task_timeout_flag = status

    def set_reset_flag(self, status: bool):
        self.reset_flag = status
