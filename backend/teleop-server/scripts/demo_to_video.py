import numpy as np
import cv2
import h5py
import argparse
from pathlib import Path


def images_to_video(
    frames: np.ndarray,
    out_path: str,
    fps: int = 30,
    codec: str = "mp4v",
    auto_bgr: bool = True,
) -> None:
    """
    Write a (n, h, w, c) NumPy array of images to an MP4 video.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (n, h, w, c).  c must be 1, 3, or 4.
        dtype may be uint8 (0-255) or float32/64 in [0, 1].

    out_path : str
        Output file path, e.g. "output.mp4".

    fps : int, default 30
        Frames per second.

    codec : str, default "mp4v"
        FourCC code. Common alternatives: "avc1" (H.264), "H264", "MJPG".

    auto_bgr : bool, default True
        If True, converts RGB input to BGR (OpenCV's default).
        Leave False if you know the array is already BGR.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected 4-D array (n,h,w,c); got {frames.shape}")

    n, h, w, c = frames.shape
    if c not in (1, 3, 4):
        raise ValueError("Channel dimension must be 1, 3, or 4")

    # Normalize dtype → uint8
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
        else:
            raise TypeError("frames must be uint8 or float32/64 in [0,1]")

    # Drop alpha if present
    if c == 4:
        frames = frames[..., :3]

    # OpenCV wants width × height
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=(c != 1))

    try:
        for i in range(n):
            frame = frames[i]
            if c == 1:  # grayscale → 3‑ch
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif auto_bgr:  # RGB → BGR
                frame = frame[..., ::-1]
            writer.write(frame)
    finally:
        writer.release()
    print(f"[✓] Saved {n} frames → {out_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Post-process HDF5 teleop data files for robomimic compatibility")

    # Get the backend directory (3 levels up from this script)
    script_path = Path(__file__).resolve()
    backend_dir = script_path.parents[3]

    default_input_file = backend_dir / "processed_data" / "dataset.hdf5"
    default_output_folder = backend_dir / "processed_data"

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(default_input_file),
        help=f"Path to input HDF5 file (default: {default_input_file})"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default=str(default_output_folder),
        help=f"Path to output folder (default: {default_output_folder})"
    )

    parser.add_argument(
        "--num_demos",
        type=int,
        default=5,
        help="Number of demos to convert (default: 5)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with h5py.File(args.dataset, "r") as f_src:
        demo_names = list(f_src["data"].keys())
        for i in range(args.num_demos):
            demo_group = f_src["data"][demo_names[i]]
            if "images" in demo_group:
                images = demo_group["images"][()]
                # Convert to video
                video_path = f"{args.output_folder}/demo_{i}.mp4"
                images_to_video(images, video_path, fps=20, codec="mp4v", auto_bgr=True)
            else:
                print("No 'images' group in demo", demo_group.name, "to build 'obs'.")
