import h5py
import json
import argparse
import numpy as np

# from scripts.post_process_data import merge_hdf5s, postprocess_hdf5

import pandas

from collections import defaultdict

EXPECTED_MESSAGE_LATENCY = 0.05


def generate_metrics(demo_dir, dst_path):

    processed_data = []
    window_size = 5

    with h5py.File(demo_dir, "r") as f_src:
        # ensure we traverse demos in order
        src_demos = list(f_src["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in src_demos])
        src_demos = [src_demos[i] for i in inds]

        for demo_ind, src_demo_key in enumerate(src_demos):
            print(f"Processing demo {demo_ind}/{len(src_demos)}")
            data = defaultdict(int)

            data["demo"] = src_demo_key

            # source group for this demonstration
            src_grp = f_src["data"][src_demo_key]

            # get which robot components are being controlled
            robots = list(src_grp["additional_data"]["teleop_commands"].keys())

            # get whether task was completed
            ep_metadata = json.loads(src_grp.attrs["metadata"])

            # Get the demo file path
            data["processed_path"] = src_grp.attrs["processed_path"]

            data["task_complete"] = ep_metadata["completed"]
            data["task"] = ep_metadata["task"]
            data["user_id"] = ep_metadata["user_id"]
            data["time"] = ep_metadata["time"]
            data["reset"] = ep_metadata["reset"]
            data["timeout"] = ep_metadata["timeout"]
            data["date"] = ep_metadata["date"]

            data["jitter_window_size"] = window_size
            data["expected_message_latency"] = EXPECTED_MESSAGE_LATENCY

            #! Get user_states
            user_states = defaultdict(dict)

            for key in robots:
                user_states[key]["obs"] = src_grp["obs"]
                user_states[key]["dpos"] = src_grp["additional_data"]["teleop_commands"][key]["position"][()]
                user_states[key]["rotation"] = src_grp["additional_data"]["teleop_commands"][key]["rotation"][()]
                user_states[key]["sent"] = src_grp["additional_data"]["timestamps"]["sent"][key][:]
                user_states[key]["received"] = src_grp["additional_data"]["timestamps"]["received"][key][:]

            robot_states = src_grp["obs"]

            counters_mapping = {val: ind for (ind, val) in enumerate(src_grp["additional_data"]["counter"][:])}

            for key in robots:
                counters = src_grp["additional_data"]["counter"][:]
                split_indices = np.where(np.diff(counters) > 1)[0] + 1
                chunks = np.split(counters, split_indices)

                for data_source in ["dpos", "rotation"]:
                    user_jitter = []
                    robot_jitter = []
                    for chunk in chunks:
                        print(f"{key} Starting from: {counters_mapping[chunk[0]]} Ending at: {counters_mapping[chunk[-1]]}")

                        if data_source == "position":
                            data_source = "dpos"  # hacky fix for now

                        # N = end_ind - start_ind + 1 = len(chunk)
                        start_ind, end_ind = counters_mapping[chunk[0]], counters_mapping[chunk[-1]]

                        user_timestamps = user_states[key]["sent"][start_ind : end_ind + 1]
                        user_timestamps_delta = replace_zeros_with_running_mean(np.diff(user_timestamps))  # N - 1 timesteps since we are calculating deltas
                        user_timestamps_avg = (user_timestamps[:-1] + user_timestamps[1:]) / 2  # N - 1 average timestamps

                        # Calculate total distance metrics based off of user data
                        user_data = user_states[key][data_source][start_ind : end_ind + 1]
                        if data_source == "rotation":
                            # Input: Absolute Rotation
                            user_deltas = get_rotation_deltas(user_data)  # N - 1 rotation deltas
                            user_speeds = user_deltas / user_timestamps_delta  # N - 1 rotation speeds
                            data[f"left_{data_source}_user_total"] += np.sum(user_deltas, axis=0)  # hackfix for now
                        else:
                            # Input: Delta Translation

                            # Throw out the first delta since we do not have the previous timestep from which it was calculated
                            user_deltas = np.linalg.norm(user_data[1:], axis=1)  # N - 1 translation deltas
                            user_speeds = (user_deltas / user_timestamps_delta)  # N - 1 translation speeds
                            data[f"left_{data_source}_user_total"] += np.sum(user_deltas, axis=0)

                        # To calculate jitter, we use a L-Smooth approach with a sliding window
                        # For each sliding window, we compute the max acceleration over that window
                        # We then take the average of these max accelerations to get the average jitter
                        for i in range(user_speeds.shape[0] - window_size + 1):
                            user_speeds_window = user_speeds[i : i + window_size]
                            user_speeds_window_delta = np.diff(user_speeds_window)

                            # We use the average timestamps to calculate the delta timestamps because the velocities are calculated from the actual timestamps
                            # Ex: v1 is calculated as (x1 - x0) / (t1 - t0) or dx / (t1 - t0) and v2 is calculated as (x2 - x1) / (t2 - t1) or dx / (t2 - t1)
                            # Then, acceleration should be calculated using the average timestamps (t0 + t1) / 2 and (t1 + t2) / 2
                            timestamp_window = user_timestamps_avg[i : i + window_size]
                            timestamp_window_delta = replace_zeros_with_running_mean(np.diff(timestamp_window))

                            user_acceleration = user_speeds_window_delta / timestamp_window_delta

                            user_jitter.append(max(user_acceleration))

                        # ########### ROBOT METRICS ############
                        # We only consider data when user is enabled since this is the only time sim is enabled
                        robot_timestamps = src_grp["additional_data"]["timestamps"]["control_loop_time"]

                        robot_timestamps_delta = replace_zeros_with_running_mean(np.diff(robot_timestamps))  # N - 1 timesteps since we are calculating deltas
                        robot_timestamps_avg = (robot_timestamps[:-1] + robot_timestamps[1:]) / 2  # N - 1 average timestamps

                        if data_source == "dpos":
                            robot_data_source = "eef_pos"  # hacky fix for now
                        elif data_source == "rotation":
                            robot_data_source = "eef_quat"

                        # Calculate total distance metrics based off of robot data
                        robot_data = robot_states[robot_data_source][start_ind : end_ind + 1]

                        if robot_data_source == "rotation":
                            # Input: Absolute Rotation
                            robot_deltas = get_rotation_deltas(robot_data)  # N - 1 rotation deltas
                            robot_speeds = robot_deltas / robot_timestamps_delta  # N - 1 rotation speeds
                            data[f"left_{robot_data_source}_robot_total"] += np.sum(robot_deltas, axis=0)
                        elif robot_data_source == "eef_quat":
                            robot_deltas = get_quat_deltas(robot_data)
                            robot_speeds = robot_deltas / robot_timestamps_delta  # N - 1 rotation speeds
                            data[f"left_{robot_data_source}_robot_total"] += np.sum(robot_deltas, axis=0)
                        else:
                            # Input: Absolute Translation
                            # We calculate deltas for translation first using np.diff
                            robot_deltas = np.linalg.norm(np.diff(robot_data, axis=0), axis=1)  # N - 1 translation deltas
                            robot_speeds = robot_deltas / robot_timestamps_delta  # N - 1 translation speeds
                            data[f"left_{robot_data_source}_robot_total"] += np.sum(robot_deltas, axis=0)

                        for i in range(robot_speeds.shape[0] - window_size + 2):
                            robot_speeds_window = robot_speeds[i : i + window_size]
                            robot_speeds_window_delta = np.diff(robot_speeds_window)

                            # We use the average timestamps to calculate the delta timestamps because the velocities are calculated from the actual timestamps
                            # Ex: v1 is calculated as (x1 - x0) / (t1 - t0) and v2 is calculated as (x2 - x1) / (t2 - t1)
                            # Then, acceleration should be calculated using the average timestamps (t0 + t1) / 2 and (t1 + t2) / 2
                            timestamp_window = robot_timestamps_avg[i : i + window_size]
                            timestamp_window_delta = replace_zeros_with_running_mean(np.diff(timestamp_window))

                            robot_acceleration = robot_speeds_window_delta / timestamp_window_delta
                            robot_jitter.append(max(robot_acceleration))

                    data[f"left_{data_source}_user_jitter_mean"] = (np.mean(user_jitter))
                    data[f"left_{data_source}_user_jitter_std"] = np.std(user_jitter)
                    data[f"left_{robot_data_source}_robot_jitter_mean"] = np.mean(robot_jitter)
                    data[f"left_{robot_data_source}_robot_jitter_std"] = (np.std(robot_jitter))

            # Calculate the network jitter coming from each component of the robot (e.g. left, right) on both the server and client side
            # Calculate the mean jitter and std of the jitters
            timestamps_grp = src_grp["additional_data"]["timestamps"]
            for key in timestamps_grp.keys():
                if key == "control_loop_time":
                    continue
                timestamps = timestamps_grp[key]
                timestampType = "server" if "received" in key else "client"
                for _, values in timestamps.items():
                    jitters = np.diff(values[:])

                    data[f"left_component_{timestampType}_network_jitter_mean (sec)"] = np.mean(jitters)
                    data[f"left_component_{timestampType}_network_jitter_std (sec)"] = np.std(jitters)

            # We calculate the network latency coming from each component of the robot (e.g. left, right)
            for key in robots:
                sent_timestamps = timestamps_grp["sent"][key][:]
                received_timestamps = timestamps_grp["received"][key][:]
                latency = received_timestamps - sent_timestamps

                data[f"left_component_network_latency_mean (sec)"] = (np.mean(latency))
                data[f"left_component_network_latency_std (sec)"] = np.std(latency)

            # We calculate the mean and std for control loop times to see how consistent the control loop is
            control_loop_time_deltas = np.diff(src_grp["additional_data"]["timestamps"]["control_loop_time"][:])
            data["control_loop_time_mean (sec)"] = np.mean(control_loop_time_deltas)
            data["control_loop_time_std (sec)"] = np.std(control_loop_time_deltas)

            processed_data.append(data)

    # Create DataFrame
    df = pandas.DataFrame(processed_data)
    print(df.head())
    df.set_index("demo", inplace=True)
    df.to_csv(dst_path)


# Function to replace zero entries with the running mean
def replace_zeros_with_running_mean(deltas):
    running_sum = 0
    running_count = 0

    for i in range(len(deltas)):
        if deltas[i] != 0:
            # Update the running sum and count
            running_sum += deltas[i]
            running_count += 1
        else:
            if running_count > 0:  # Avoid division by zero
                deltas[i] = running_sum / running_count
                print(f"Warning: Zero entry encountered in deltas. Replacing with running mean: {deltas[i]}.")
            else:
                deltas[i] = EXPECTED_MESSAGE_LATENCY  # If the first two timestamps are the same, we assume the expected message latency
                print(f"Warning: Zero entry encountered in first two deltas. Replacing with expected message latency: {EXPECTED_MESSAGE_LATENCY}.")

    return deltas


def get_rotation_deltas(rotation_matrices):
    """
    Returns a list of size N - 1 where each element is the angle between the rotation matrices at index i and i + 1
    """
    rotation_deltas = []
    for i in range(len(rotation_matrices) - 1):
        # Compute the relative rotation matrix
        R_i = rotation_matrices[i]
        R_next = rotation_matrices[i + 1]
        R_relative = np.dot(R_i.T, R_next)

        # Compute the rotation angle
        trace = np.trace(R_relative)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))

        rotation_deltas.append(angle)

    return rotation_deltas


def get_quat_deltas(quaternions):
    """
    Returns a list of size N - 1 where each element is the angle between the quaternions at index i and i + 1
    """
    rotation_deltas = []
    for i in range(len(quaternions) - 1):
        q_i = quaternions[i]
        q_next = quaternions[i + 1]

        # Compute the rotation angle
        angle = 2 * np.arccos(np.clip(abs(np.dot(q_i, q_next)), -1, 1))

        rotation_deltas.append(angle)

    return rotation_deltas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # location of folder to save hdf5 files in
    parser.add_argument(
        "--input_file",
        type=str,
        required=True
    )

    # location of source folder with hdf5s to merge / postprocess
    parser.add_argument(
        "--output_file",
        type=str,
        required=True
    )

    args = parser.parse_args()

    # # create save dir if it doesn't exist
    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)

    # if args.demo_dir is not None:
    #     # merge hdf5s into one source hdf5 first (with sorted demos)
    #     merge_hdf5s(demo_dir=args.demo_dir, save_dir=args.save_dir)

    # # paths to source and postprocessed hdf5s
    # source_hdf5_path = os.path.join(args.save_dir, "source.hdf5")
    # assert os.path.exists(source_hdf5_path)

    # # postprocess -- KEEPING FAILURES so that we can return accuracy metrics
    # postprocess_hdf5(src_hdf5_path=source_hdf5_path, dst_hdf5_path=dest_hdf5_path, keep_failures=True)

    dst_metrics_path = args.output_file
    generate_metrics(demo_dir=args.input_file, dst_path=dst_metrics_path)
