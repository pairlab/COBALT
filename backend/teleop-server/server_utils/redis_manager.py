import json
from typing import Any, Dict, List

import redis
from .device_state import DeviceState
from .session import Session

from data_collection.data_collector import DataCollector
from controllers.controller import make_controller
from robots.interfaces.teleop_robot import make_robot


class RedisManager:
    """Manages session lifecycle and data flow using Redis."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0) -> None:
        """
        Initialize Redis connection.

        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
        """
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )
        self.sessions: Dict[str, Session] = {}
        self.env_sessions_map: Dict[int, str] = {}
        self.available_environments: List[int] = []

    def get_session(self, session_id: str) -> Session:
        """
        Retrieve a session by ID.

        Args:
            session_id: Target session ID.

        Returns:
            Session instance.
        """
        if not session_id:
            return None

        return self.sessions.get(session_id, None)

    def get_pipeline(self) -> Any:
        """
        Create a Redis pipeline.

        Returns:
            Redis pipeline instance.
        """
        return self.redis_client.pipeline()

    def remove_from_redis_pending_add(self, session_id: str, pipeline: Any) -> None:
        """
        Remove a session from the pending add queue.

        Args:
            session_id: Target session ID.
        """
        pipeline.lrem("pending_add", 1, session_id)
        pipeline.srem("pending_add_set", session_id)

    def remove_from_redis_pending_delete(self, session_id: str, pipeline: Any) -> None:
        """
        Remove a session from the pending delete queue.

        Args:
            session_id: Target session ID.
        """
        pipeline.lrem("pending_delete", 1, session_id)
        pipeline.srem("pending_delete_set", session_id)

    def remove_from_redis(self, session_id: str, pipeline: Any) -> None:
        """
        Remove a session from Redis.

        Args:
            session_id: Target session ID.
        """
        pipeline.delete(f"sessions:{session_id}")
        pipeline.delete(f"sessions-{session_id}:images")

    def reject_invalid_session(self, session_id: str, reason: str, pipeline: Any) -> None:
        """Reject a malformed/incompatible session.

        We do not delete Redis session keys here because websocket-server owns
        client socket lifecycle and closes sockets when it observes invalid status.
        """
        print(f"Rejecting session {session_id}: {reason}")
        if session_id in self.sessions:
            self._delete_session(session_id)

        pipeline.hset(
            f"sessions:{session_id}",
            mapping={"status": -1, "error": reason},
        )
        self.remove_from_redis_pending_add(session_id, pipeline=pipeline)

    def process_delete(self, session_id: str, pipeline: Any) -> None:
        """
        Process session deletion.

        Args:
            session_id: Target session ID.
        """
        if session_id in self.sessions:
            self._delete_session(session_id)
            print(f"Deleted session {session_id}")
        else:
            # Session not found, remove from pending queue
            print(f"Session {session_id} not found in Locally stored sessions")

        self.remove_from_redis_pending_delete(session_id, pipeline=pipeline)
        self.remove_from_redis(session_id, pipeline=pipeline)

    def process_add(self, session_id: str, simulator, config, pipeline: Any) -> None:

        # Safety check to make sure that the session is initialized with the exact number of devices
        deviceCount = self.redis_client.hget(f"sessions:{session_id}", "device_count")
        if deviceCount is None:
            self.reject_invalid_session(
                session_id,
                "missing device_count in Redis",
                pipeline=pipeline,
            )
            return None

        if int(deviceCount) != config.num_device:
            self.reject_invalid_session(
                session_id,
                (
                    "invalid device count "
                    f"({deviceCount} != expected {config.num_device})"
                ),
                pipeline=pipeline,
            )
            return None

        if session_id not in self.sessions:
            if len(self.available_environments) > 0:
                try:
                    env_idx = self.available_environments.pop()
                    self._create_session(
                        session_id,
                        env_idx,
                        config,
                        simulator,
                        pipeline=pipeline,
                    )
                    self.remove_from_redis_pending_add(session_id, pipeline=pipeline)
                except TypeError as e:
                    print(f"Error creating session {session_id}: {e}")
                    # Tried to fetch session id from redis but it is not present
                    self.remove_from_redis_pending_add(session_id, pipeline=pipeline)
                    print(f"Session {session_id} not found in Redis")

                    # Add environment back to available list
                    self.available_environments.append(env_idx)

                except IndexError as e:
                    # No available environments do not remove from pending queue
                    print(f"No available environments for session {session_id}, Error: {e}")
            else:
                # No available environments
                print(f"Session {session_id} has no available environments, waiting for next update")
        else:
            # Session already exists
            print(f"Session {session_id} already exists")

            self.remove_from_redis_pending_add(session_id, pipeline=pipeline)

    def update_sessions(self, config: Any, simulator: Any) -> None:
        """
        Update session list based on Redis queues.

        Args:
            available_environments: List of free environment indices.
            config: System configuration.
            robot: Robot interface instance.

        Returns:
            Updated list of available environments.
        """
        # Create a pipeline to batch process Redis updates
        redis_updates = self.get_pipeline()

        # Process deletions
        pending_delete = self.redis_client.lrange("pending_delete", 0, -1)

        list(
            map(
                lambda session_id: self.process_delete(
                    session_id, pipeline=redis_updates
                ),
                pending_delete,
            )
        )

        # Process additions
        number_to_fetch = len(self.available_environments)
        pending_add = self.redis_client.lrange("pending_add", 0, number_to_fetch)

        list(
            map(
                lambda session_id: self.process_add(
                    session_id,
                    simulator,
                    config,
                    pipeline=redis_updates,
                ),
                pending_add,
            )
        )

        redis_updates.execute()

    def _create_session(
        self,
        session_id: str,
        env_idx: int,
        config: Any,
        simulator: Any,
        pipeline: Any,
    ) -> None:
        """
        Create and register a new session.

        Args:
            session_id: Unique session identifier.
            env_idx: Assigned environment index.
            config: Configuration settings.
            robot: Robot interface instance.
        """
        session_config = json.loads(self.redis_client.hget(f"sessions:{session_id}", "config"))

        devices = {}
        for device_id, device_cfg in session_config["devices"].items():
            devices[device_id] = DeviceState(device_id, device_cfg)

        user_id = session_config.get("username", "no_user_id_in_config")

        data_collector = (
            DataCollector(config, simulator, user_id, env_id=env_idx)
            if config.data_collection.enabled
            else None
        )

        # Create robot interface (e.g. RobosuiteRobot, YAMRobot)
        robot = make_robot(config.robot.type, config=config)

        # Create controller interface (e.g. OSCController)
        controller = make_controller(
            controller_name=config.controller.type,
            config=config,
            robot=robot,
            env_idx=env_idx,
        )

        cache_image_data = simulator.get_image_data()

        self.sessions[session_id] = Session(
            session_id,
            config,
            devices,
            data_collector,
            controller,
            env_idx,
            cache_image_data[env_idx] if cache_image_data is not None else None,
        )
        self.env_sessions_map[env_idx] = session_id
        print(f"Created session {session_id} with environment {env_idx}")

        pipeline.hset(f"sessions:{session_id}", mapping={"status": 1})

    def _delete_session(self, session_id: str) -> None:
        """
        Delete a session and free its resources.

        Args:
            session_id: Session to delete.
            available_environments: List to append freed environment index.
        """
        session = self.sessions.pop(session_id, None)
        if session:
            self.available_environments.append(session.env_idx)
            self.env_sessions_map.pop(session.env_idx)

            session.end_trial(success=False, reset=False, timeout=False, disconnected=True)
            session.stop_video_stream()

    def get_active_sessions_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch and update data for all active sessions.

        Returns:
            Dictionary mapping session IDs to parsed client data.
        """
        session_data = {}

        # Skip processing if no active sessions
        if not self.sessions:
            return session_data

        # Create pipeline and queue all hgetall commands
        pipeline = self.redis_client.pipeline()
        for session_id in self.sessions.keys():
            session = self.get_session(session_id)
            if session is None:
                continue
            else:
                devices = session.devices
                pipeline.hmget(f"sessions:{session_id}", devices)

        # Execute pipeline in a single network round-trip
        results = pipeline.execute()

        # Process results
        for (session_id, session), data in zip(self.sessions.items(), results):
            if None not in data:
                try:
                    client_data = session.parse_message(data)
                    session.update(client_data)
                    session_data[session_id] = client_data
                except KeyError:
                    # Session data not found in Redis
                    print(f"Session {session_id} data not found in Redis")
                    continue

        return session_data

    def send_stream_data(self, stream_data: Any) -> None:
        for _, session in self.sessions.items():
            session.send_frames(stream_data[session.env_idx])

    def send_response(self, session_id: str, pipeline: Any) -> None:
        """Send response data to the client via Redis.

        Args:
            session_id: Target session ID.
            observation: Observation data.
            reward: Reward value.
            done: Completion flag.
        """
        session = self.get_session(session_id)

        if session is None:
            return

        response_data = self.sessions[session_id].get_response()

        device_response = {
            f"response:{device_id}": json.dumps(response_data[device_id])
            for device_id in response_data
        }
        device_response["done"] = int(session.get_is_done())
        pipeline.hset(
            f"sessions:{session_id}",
            mapping=device_response,
        )
