# #!/usr/bin/env python

import argparse
import sys
import time
import signal
import json
import asyncio
import websockets
import urllib.parse
import traceback
# import ntplib

from configs.config import make_config


class TeleoperationClient(object):
    """
    This class handles the logic for receiving controls from the
    keyboard and virtual reality interfaces and sending the controls
    over a WebSocket connection to the server.
    """

    def __init__(self, config, args):
        self.config = config
        self.args = args
        # time interval (in seconds) for sampling the VR controller.
        self.time_interval = config.client.sample_interval

        # Setup a connection to the client interface.
        self.clients = []
        for controller in self.config.client.controllers:
            sw = controller[0]
            print("setting up controller", sw)
            if sw == "keyboard":
                from clients.keyboard import KeyboardClient

                client = KeyboardClient()
            elif sw == "3d_mouse":
                from clients.spacenavigator import SpaceNavigator

                client = SpaceNavigator()
            else:
                raise Exception("Pick a valid client, " + str(repr(sw)) + "is not acceptable.")
            self.clients.append(client)

        print("set up controllers")

        # this indicator is toggled to indicate that a server reset has taken place
        self.last_reset_indicator = None

        # this signal is driven high to indicate successful task completion
        self.last_task_completion_status = False
        self.last_task_timeout_status = False

        # Lock for thread-safe access to client objects shared between send/receive tasks
        self._client_lock = asyncio.Lock()

        # WebSocket connection
        self.websocket = None
        self.connection_ready = False

        # Build WebSocket URL
        self.ws_url = self._build_websocket_url()
        print("WebSocket URL:", self.ws_url)

        # Register a handler to handle SIGINTs (Ctrl-C) so we can clean up the connection.
        signal.signal(signal.SIGINT, self._sigint_handler)

        self.time_offset = self.calculate_time_offset()

        # Run the client main loop.
        asyncio.run(self.run())

    def _build_websocket_url(self):
        """
        Build WebSocket URL from configuration.
        """
        # Build query parameters
        query_params = []

        # Add any additional configuration parameters
        if hasattr(self.config, 'sim'):
            config_data = {
                "sim": getattr(self.config.sim, 'sim', 'default'),
                "arm": getattr(self.config.sim, 'arm', 'right'),
                "sim_type": getattr(self.config.sim, 'sim_type', 'single'),
                "username": getattr(self.config, 'username', 'default')
            }
            query_params.append(f"config={urllib.parse.quote(json.dumps(config_data))}")

        # Add device_id and session_id if available
        if hasattr(self.config, 'device_id') and self.config.device_id:
            query_params.append(f"device_id={self.config.device_id}")
        if hasattr(self.args, 'session_id') and self.args.session_id:
            query_params.append(f"session_id={self.args.session_id}")

        # Build the full URL
        if self.args == 'test':
            url = f"ws://{self.args.host}:{self.args.port}/ws"
        else:
            url = f"ws://{self.args.host}:{self.args.port}/{self.args.task}/ws"

        if query_params:
            url += "?" + "&".join(query_params)

        return url

    def _sigint_handler(self, signal, frame):
        """
        Handler for SIGINT. Properly close the WebSocket connection and handle any other
        needed cleanup here.
        """
        print("\nClient terminated.")
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        sys.exit()

    async def send_message(self, msg):
        """
        Helper function to send a message over WebSocket.

        :param msg: The message data to send.

        :return: True if message was sent successfully, False otherwise.
        """
        try:
            if self.websocket:
                await self.websocket.send(json.dumps(msg))
                return True
            else:
                print("WebSocket not connected")
                return False
        except Exception as e:
            print(f"WebSocket send error: {e}")
            return False

    async def recv_json(self):
        """
        Receive a JSON message from the WebSocket server.

        :return: The parsed JSON message, or None if error.
        """
        try:
            if self.websocket:
                message = await self.websocket.recv()
                return json.loads(message)
            else:
                return None
        except websockets.ConnectionClosed:
            print("WebSocket connection closed by server.")
            return None
        except Exception as e:
            print(f"WebSocket receive error: {e}")
            return None

    async def run(self):
        """
        Client main loop using WebSocket. Establish connection, handle handshake,
        then start separate tasks for sending and receiving data.
        """
        try:
            print(f"Connecting to WebSocket: {self.ws_url}")

            # Establish WebSocket connection
            async with websockets.connect(self.ws_url) as websocket:
                self.websocket = websocket
                print("Connected to WebSocket server.")

                # Wait for initial acknowledgement
                message = await self.recv_json()
                if message:
                    print(f"Server: {message}")
                    if message.get("type") == "error":
                        raise Exception(message.get("data", "Unknown error"))

                # Wait for status signal
                message = await self.recv_json()
                if message and message.get("type") == "status":
                    if not message.get("data", {}).get("ready", False):
                        raise Exception("Simulation failed to start.")
                    print("Server ready.")
                    self.connection_ready = True

                # Start separate tasks for sending and receiving
                send_task = asyncio.create_task(self._send_loop())
                recv_task = asyncio.create_task(self._receive_loop())

                # Wait for either task to complete (which indicates an error or shutdown)
                try:
                    await asyncio.gather(send_task, recv_task, return_exceptions=True)
                except Exception as e:
                    print(f"Error in main communication tasks: {e}")
                finally:
                    # Cancel any remaining tasks
                    if not send_task.done():
                        send_task.cancel()
                    if not recv_task.done():
                        recv_task.cancel()

        except Exception as e:
            print(f"WebSocket error: {e}. Retrying in 3 seconds...")
            self.connection_ready = False
            self.websocket = None
            await asyncio.sleep(3)

        print("WebSocket client shutting down.")

    async def _send_loop(self):
        """
        Continuous loop for sending controller data to the server.
        """
        message_id = 0

        while self.connection_ready:
            cur = time.time()

            try:
                # query each controller for their updated state
                controller_states = []
                async with self._client_lock:
                    for client in self.clients:
                        controller_state = client.get_controller_state()
                        controller_states.append(controller_state)

                # preserve binary compatibility, allow older code to work
                if len(controller_states) == 1:
                    controller_states = controller_states[0]

                # Convert controller state to WebSocket message format
                device_data = self._convert_controller_state_to_device_data(controller_states, message_id)

                # Create WebSocket message
                websocket_message = {
                    "type": "device data",
                    "data": device_data
                }

                # Send the message
                success = await self.send_message(websocket_message)

                if not success:
                    print("Failed to send message, breaking connection")
                    self.connection_ready = False
                    break

                # enforce publishing rate
                diff = time.time() - cur
                await asyncio.sleep(max(self.time_interval - diff, 0.0))
                message_id += 1

            except Exception as e:
                print(f"Error in send loop: {e}")
                self.connection_ready = False
                break

    async def _receive_loop(self):
        """
        Continuous loop for receiving and processing server responses.
        """
        while self.connection_ready:
            try:
                # Wait for and process server response
                response = await self.recv_json()
                if response is None:
                    print("Failed to receive response, breaking connection")
                    self.connection_ready = False
                    break

                # Process server response (maintain compatibility with original logic)
                await self._process_server_response(response)

            except Exception as e:
                print(f"Error in receive loop: {e}")
                traceback.print_exc()
                self.connection_ready = False
                break

    def _convert_controller_state_to_device_data(self, controller_state, message_id):
        """
        Convert controller state to the device data format expected by the WebSocket server.
        This tries to maintain compatibility with the original TCP format while adapting
        to the WebSocket message structure seen in server_load_tester.py.
        """
        # Handle single controller state (preserve binary compatibility)
        if isinstance(controller_state, dict):
            state = controller_state
        else:
            # Multiple controllers - for now, use the first one
            state = controller_state[0] if controller_state else {}

        # Extract common fields with defaults
        device_data = {
            "reset": state.get("reset", 0),
            "completion": state.get("completion", 0),
            "timeout": state.get("timeout", 0),
            "enable": state.get("engage", 0),
            "id": message_id,
            "valid": state.get("valid", 1),
            "timestamp": time.time() + self.time_offset,
        }

        # Handle different controller types and their specific data
        if "dpos" in state:
            device_data["dpos"] = state["dpos"]
        elif "translation" in state:
            device_data["dpos"] = state["translation"]
        else:
            device_data["dpos"] = [0.0, 0.0, 0.0]

        if "rotation" in state:
            device_data["rotation"] = state["rotation"]
        elif "quaternion" in state:
            # Convert quaternion to rotation matrix if needed
            # For now, just pass the quaternion
            device_data["rotation"] = state["quaternion"]
        else:
            device_data["rotation"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        if "grasp" in state:
            device_data["grasp"] = state["grasp"]
        elif "grip" in state:
            device_data["grasp"] = state["grip"]
        else:
            device_data["grasp"] = 0

        return device_data

    async def _process_server_response(self, response):
        """
        Process the server response for handling reset, completion, and timeout signals.
        Uses async lock to protect client object modifications from concurrent access.
        """
        # The exact format may vary depending on the server implementation
        data = response.get("data", {})

        # Convert string to dictionary
        data = json.loads(data)

        # Try to extract the 3-character status similar to the original TCP format
        # This may need adjustment based on actual server response format
        reset_indicator = data.get("reset", 0)
        completion_status = data.get("complete", 0)
        timeout_status = data.get("timeout", 0)

        # Alternative: if the server sends a simple status string like the original
        if "status" in data and isinstance(data["status"], str) and len(data["status"]) >= 3:
            status_str = data["status"]
            reset_indicator = int(status_str[0])
            completion_status = int(status_str[1])
            timeout_status = int(status_str[2])

        # Use async lock to protect client object modifications
        async with self._client_lock:
            # check for reset acknowledgment
            if (self.last_reset_indicator is not None) and (self.last_reset_indicator != reset_indicator):
                # drive reset signal low so that client can provide another low to high
                # transition if they want to reset the server
                for client in self.clients:
                    client.set_reset_state(0)
            self.last_reset_indicator = reset_indicator

            # check for task completion indication (low to high transition)
            if not self.last_task_completion_status and bool(completion_status):
                for client in self.clients:
                    client.handle_task_completion()
            self.last_task_completion_status = bool(completion_status)

            # check for task timeout indication (low to high transition)
            if not self.last_task_timeout_status and bool(timeout_status):
                for client in self.clients:
                    client.handle_task_timeout()
            self.last_task_timeout_status = bool(timeout_status)

    def calculate_time_offset(self):
        """
        Calculate the time offset between client and server.
        """
        # total_offset = 0
        # numRequests = 10
        # actualRequests = 0

        # for _ in range(numRequests):
        #     c = ntplib.NTPClient()
        #     try:
        #         response = c.request('time.google.com')
        #         total_offset += response.offset
        #         actualRequests += 1
        #     except Exception as e:
        #         print(f"NTP request error: {e}")
        #         continue
        #     time.sleep(0.05)

        # time_offset = total_offset / actualRequests if actualRequests > 0 else 0
        # print("Network Clock is ahead of the System clock by: %.5f seconds" % time_offset)

        time_offset = 0  # For now, we can set this to 0 or implement a more robust time sync if needed

        return time_offset


if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser(description='Run a teleoperation client to control a robot.')
    parser.add_argument('--config', default='BaseClientConfig', help='name of config class to use')
    parser.add_argument('--host', default='localhost', help='WebSocket server host')
    parser.add_argument('--port', default=8080, type=int, help='WebSocket server port')
    parser.add_argument('--task', default='test', help='Task name')
    parser.add_argument('--session_id', default=None, help='Session ID')
    parser.add_argument('--sim-type', default='bimanual', help='Simulation type (e.g. single, bimanual)')
    args = parser.parse_args()

    client_config = make_config(args.config)

    # Manual overrides from command line arguments (if provided)
    if hasattr(args, 'sim_type'):
        setattr(client_config.sim, 'sim_type', args.sim_type)

    print(client_config)

    client = TeleoperationClient(config=client_config, args=args)
