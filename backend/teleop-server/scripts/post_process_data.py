import os
import glob
import h5py
import numpy as np
import argparse
from pathlib import Path


def is_chunk_group(h5group):
    """
    Returns True if every key in the group is a numeric string.
    Assumes such a group holds time-series chunks.
    """
    return all(key.isdigit() for key in h5group.keys())


def merge_chunk_group(src_group):
    """
    Given a group whose keys are chunk identifiers (like "000000", "000001"),
    sorts them and concatenates the datasets along axis 0.
    """
    chunk_keys = sorted(src_group.keys(), key=lambda k: int(k))
    arrays = [src_group[ckey][()] for ckey in chunk_keys]
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def merge_group(src_group, tgt_group):
    """
    Recursively copy groups/datasets from src_group to tgt_group.
    If a group is a chunk group (keys are numeric), merge its chunks.
    """
    for key in src_group:
        item = src_group[key]
        if isinstance(item, h5py.Group):
            # If the group contains only numeric keys, merge it.
            if is_chunk_group(item):
                merged_array = merge_chunk_group(item)
                tgt_group.create_dataset(key, data=merged_array)
            else:
                new_group = tgt_group.create_group(key)
                merge_group(item, new_group)
        elif isinstance(item, h5py.Dataset):
            # If it is a dataset, copy it directly.
            src_group.copy(key, tgt_group)


def postprocess_demo(demo_group):
    """
    Restructure the demo group to adhere to Robomimic conventions.

    1. Ensure a top-level "actions" dataset exists.
       If not, copy it from "additional_data/robot_controls/actions".

    2. Create an "obs" group.
       We assume that the merged "states" group holds the per-timestep observation
       data. For each observation modality found in the first chunk of "states",
       the function iterates through all chunks, concatenates the arrays,
       and writes them as a dataset in "obs".
    """
    # --- Ensure top-level actions ---
    if "actions" not in demo_group:
        try:
            # Copy the merged actions dataset from its nested location.
            actions_ds = demo_group["additional_data/robot_controls/actions"]
            demo_group.copy(actions_ds.name, demo_group, "actions")
        except Exception as e:
            print(
                "Could not find actions in additional_data/robot_controls/actions:", e
            )

    # --- Build the obs group from states ---
    if "obs" not in demo_group:
        obs_group = demo_group.create_group("obs")
        if "states" in demo_group:
            states_group = demo_group["states"]
            state_keys = list(states_group.keys())
            if len(state_keys) == 0:
                print(
                    "No datasets found in the states group for demo:", demo_group.name
                )
            else:
                for key in state_keys:
                    # Merge the datasets in the states group.
                    merged_array = states_group[key][()]
                    # Create a new dataset in the obs group.
                    obs_group.create_dataset(key, data=merged_array)
        else:
            print("No 'states' group in demo", demo_group.name, "to build 'obs'.")

    # Add num_samples attribute to the demo group.
    demo_group.attrs["num_samples"] = len(demo_group["actions"])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Post-process HDF5 teleop data files for robomimic compatibility")

    # Get the backend directory (3 levels up from this script)
    script_path = Path(__file__).resolve()
    backend_dir = script_path.parents[3]

    default_input_folder = backend_dir / "teleop_data"
    default_output_file = backend_dir / "processed_data" / "dataset.hdf5"

    parser.add_argument(
        "--input_folder",
        type=str,
        default=str(default_input_folder),
        help=f"Path to folder containing input HDF5 files (default: {default_input_folder})"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=str(default_output_file),
        help=f"Path to output merged HDF5 file (default: {default_output_file})"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Set your input folder and output file path ---
    input_folder = args.input_folder
    output_file = args.output_file

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # --- Get list of all HDF5 files in the folder ---
    file_list = glob.glob(os.path.join(input_folder, "*.hdf5"))
    if not file_list:
        print("No HDF5 files found in the specified folder.")
        return

    # --- Create (or overwrite) the output file ---
    with h5py.File(output_file, "w") as out_f:
        # Create the top-level "data" group.
        data_out = out_f.create_group("data")

        demo_counter = 0  # To provide unique demo names.

        # Loop over all source files.
        for file_path in file_list:
            with h5py.File(file_path, "r") as in_f:
                if "data" not in in_f:
                    print(f"File {file_path} lacks a 'data' group; skipping.")
                    continue

                # Process each demonstration in the source file.
                for demo_key in in_f["data"]:
                    src_demo = in_f["data"][demo_key]
                    tgt_demo_name = f"demo_{demo_counter}"
                    demo_counter += 1
                    tgt_demo = data_out.create_group(tgt_demo_name)

                    # Merge all subgroups and datasets.
                    merge_group(src_demo, tgt_demo)

                    tgt_demo.attrs["processed_path"] = file_path

                    # Postprocess the demo for Robomimic compatibility.
                    postprocess_demo(tgt_demo)

                    for attr in in_f["data"]["demo_0"].attrs:
                        if attr != "num_demos":
                            tgt_demo.attrs[attr] = in_f["data"]["demo_0"].attrs[attr]

                    print(f"Merged demo from '{file_path}' (source group: {demo_key}) into '{tgt_demo_name}'.")

        data_out.attrs["num_demos"] = demo_counter
    print(f"All demos merged into output file: {output_file}")


if __name__ == "__main__":
    main()
