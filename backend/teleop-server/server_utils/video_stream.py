import av
import redis
import multiprocessing
import numpy as np
import json
import time
import base64
from multiprocessing import Queue
from queue import Empty, Full


class VideoStream(multiprocessing.Process):
    def __init__(self, config, session_id: str):
        super().__init__()
        self.session_id = session_id

        # Video configuration
        self.video_codec = config.video.codec
        self.video_fps = config.video.fps
        self.default_width = config.video.width
        self.default_height = config.video.height

        self.codec_ctx_by_view = {}
        self.frame_index_by_view = {}
        self.active_views = set()

        # Allow h264, vp8, vp9, and h264_nvenc codecs
        assert config.video.codec in [
            "h264",
            "vp8",
            "vp9",
            "h264_nvenc",
        ], "Unsupported codec"

        # Redis configuration
        self.buffer_size = config.video.buffer_size
        self.redis = redis.Redis(
            host=config.redis.hostname, port=config.redis.port, db=config.redis.db
        )

        # Use a Queue to buffer frames instead of a single shared variable.
        self.frame_queue = Queue(maxsize=self.buffer_size)
        self.i = 0
        self.running = True
        self.default_view_name = "primary"

    def _get_codec_options(self):
        if self.video_codec == "h264_nvenc":
            return {"preset": "p1", "tune": "ull", "g": "5"}
        if self.video_codec == "vp8":
            return {
                "deadline": "realtime",
                "cpu-used": "5",
                "g": "30",
            }
        if self.video_codec == "vp9":
            return {
                "deadline": "realtime",
                "cpu-used": "5",
                "lag-in-frames": "0",
                "g": "30",
            }
        return {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "g": "30",
        }

    def _get_or_create_codec_ctx(self, view_name: str, frame: np.ndarray):
        if view_name in self.codec_ctx_by_view:
            return self.codec_ctx_by_view[view_name]

        codec = av.Codec(self.video_codec, "w")
        codec_ctx = codec.create()
        height, width = frame.shape[:2]
        codec_ctx.width = width if width > 0 else self.default_width
        codec_ctx.height = height if height > 0 else self.default_height
        codec_ctx.framerate = self.video_fps
        codec_ctx.pix_fmt = "yuv420p"
        codec_ctx.options = self._get_codec_options()

        self.codec_ctx_by_view[view_name] = codec_ctx
        self.frame_index_by_view.setdefault(view_name, 0)
        self.active_views.add(view_name)

        # Publish available view names for this session (read by media server).
        self.redis.sadd(f"sessions-{self.session_id}:image_views", view_name)
        return codec_ctx

    def run(self):
        print(f"Starting video stream for session {self.session_id}")

        while self.running:
            try:
                frames = self.frame_queue.get(timeout=1)
            except Empty:
                continue

            # Use a sentinel value (None) to signal shutdown.
            if frames is None:
                break

            for view_name, view_frame in frames.items():
                if view_frame is None:
                    continue

                codec_ctx = self._get_or_create_codec_ctx(view_name, view_frame)

                # Encode and stream the frame
                av_frame = av.VideoFrame.from_ndarray(view_frame, format="rgb24")
                packets = codec_ctx.encode(av_frame)
                for packet in packets:
                    packet_bytes = bytes(packet)
                    packet_b64 = base64.b64encode(packet_bytes).decode("utf-8")
                    now = time.time()  # Unix epoch timestamp in seconds
                    frame_payload = {
                        "frame": packet_b64,
                        "index": self.frame_index_by_view[view_name],
                        "ts": now,
                    }

                    view_list_key = f"sessions-{self.session_id}:images:{view_name}"
                    self.redis.rpush(view_list_key, json.dumps(frame_payload))
                    self.redis.ltrim(view_list_key, -self.buffer_size, -1)

                self.frame_index_by_view[view_name] += 1

            self.i += 1

        # Flush any remaining packets and close
        for view_name, codec_ctx in self.codec_ctx_by_view.items():
            try:
                codec_ctx.encode()
            except Exception:
                pass
            if codec_ctx.is_open:
                codec_ctx.close()

        self.redis.delete(f"sessions-{self.session_id}:image_views")
        for view_name in self.active_views:
            self.redis.delete(f"sessions-{self.session_id}:images:{view_name}")
        self.redis.close()
        print(f"Stopping video stream for session {self.session_id}")

    def update_frame(self, frames: dict):
        # Non-blocking put; drop the frame if the queue is full to maintain low latency.
        try:
            self.frame_queue.put_nowait(frames)
        except Full:
            print("Frame queue full, dropping frame.")

    def stop(self):
        self.running = False
        # Push a sentinel value to ensure the thread exits the blocking get.
        self.frame_queue.put(None)
        self.join()
