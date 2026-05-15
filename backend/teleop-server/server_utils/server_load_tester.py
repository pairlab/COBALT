#!/usr/bin/env python3

import asyncio
import websockets
import json
import random
import argparse
import time
import urllib.parse
import multiprocessing
import socket


def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        # Connect to a remote address to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


# WebSocket server URL
SERVER_URL = f"ws://{get_local_ip()}:8080/liftCube/ws"


async def recv_json(websocket):
    """Receive a JSON message from the WebSocket server."""
    message = await websocket.recv()
    return json.loads(message)


async def receive_messages(websocket, device_id):
    """Continuously receive messages from the server in a separate task."""
    while True:
        try:
            message = await recv_json(websocket)
        except websockets.ConnectionClosed:
            print(f"[{device_id}] Connection closed by server.")
            break
        except Exception as e:
            print(f"[{device_id}] Receive error: {e}")
            break


async def send_pose_data(sim, sim_type, arm, device_id=None, session_id=None):
    """Simulate a client sending pose data over WebSocket."""
    while True:
        try:
            # Build the connection URL with query parameters
            query_params = []
            if device_id:
                query_params = [f"device_id={device_id}"]
            if session_id:
                query_params.append(f"session_id={session_id}")
            config = {"sim": sim, "arm": arm, "sim_type": sim_type}
            query_params.append(f"config={urllib.parse.quote(json.dumps(config))}")
            url = f"{SERVER_URL}?" + "&".join(query_params)
            print(f"[{device_id}] Connecting to {url}")

            # Establish WebSocket connection
            async with websockets.connect(url) as websocket:
                print(f"[{device_id}] Connected to server.")

                # Wait for initial acknowledgement
                message = await recv_json(websocket)
                print(f"[{device_id}] Server: {message}")
                if message["type"] == "error":
                    raise Exception(message["data"])

                # Wait for ready signal
                message = await recv_json(websocket)
                if message["type"] == "ready" and not message["data"]["ready"]:
                    raise Exception("Simulation failed to start.")
                print(f"[{device_id}] Server ready.")

                # Start receiving messages in a separate task
                receive_task = asyncio.create_task(receive_messages(websocket, device_id))

                # Send pose data every 50ms
                i = 0
                step_size = 1e-20
                grasp = 0
                while True:
                    start_time = time.time()

                    # Generate random pose data
                    dpos = [random.uniform(-step_size, step_size) for _ in range(3)]
                    rotation = [random.uniform(-step_size, step_size) for _ in range(9)]
                    if random.random() < 0.05:  # 5% chance to toggle grasp
                        grasp ^= 1
                    reset = 1 if random.random() < 0.005 else 0  # 0.5% chance of reset

                    pose_data = {
                        "type": "device data",
                        "data": {
                            "reset": reset,
                            "completion": 0,
                            "timeout": 0,
                            "enable": 1,
                            "id": i,
                            "dpos": dpos,
                            "rotation": rotation,
                            "grasp": grasp,
                            "valid": 1,
                            "timestamp": time.time(),
                        },
                    }

                    # Send data to the server
                    await websocket.send(json.dumps(pose_data))

                    # Maintain 50ms interval
                    elapsed_time = time.time() - start_time
                    await asyncio.sleep(max(0, 0.05 - elapsed_time))
                    i += 1

        except Exception as e:
            print(f"[{device_id}] Error: {e}. Retrying in 3 seconds...")
            await asyncio.sleep(3)
        finally:
            if "receive_task" in locals():
                receive_task.cancel()
                try:
                    await receive_task
                except asyncio.CancelledError:
                    pass


def client_process(sim, sim_type, arm, device_id, session_id):
    """Run the asyncio event loop for a single client in a process."""
    asyncio.run(send_pose_data(sim, sim_type, arm, device_id, session_id))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load test with multiprocessing.")
    parser.add_argument("--sim", type=str, default="robosuite", help="Simulation name")
    parser.add_argument("--sim_type", type=str, default="bimanual", help="Simulation type")
    parser.add_argument("--arm", type=str, default="left", help="Arm selection")
    parser.add_argument("--device_id", type=str, default=None, help="Device ID")
    parser.add_argument("--session_id", type=str, default=None, help="Session ID")
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients")
    return parser.parse_args()


def main():
    """Launch multiple client processes."""
    args = parse_args()
    if args.device_id == "None":
        args.device_id = None
    if args.session_id == "None":
        args.session_id = None

    sim = args.sim
    sim_type = args.sim_type
    arm = args.arm
    session_id = args.session_id
    device_id = args.device_id
    num_clients = args.num_clients

    # Start processes
    processes = []
    for i in range(num_clients):
        p = multiprocessing.Process(
            target=client_process,
            args=(sim, sim_type, arm, device_id, session_id),
            daemon=True,
        )
        processes.append(p)
        p.start()

    # Wait for processes (runs until interrupted)
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
