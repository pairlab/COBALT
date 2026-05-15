"""
Some utility functions.

NOTE: convention for quaternions is (x, y, z, w).
"""

import functools
import json
import math
import sys
import time
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Union
import psutil
import pynvml
import torch

import h5py
import numpy as np

pi = np.pi
EPS = np.finfo(float).eps * 4.0
np.set_printoptions(precision=5)

# use numba on python 3 by default and not on python 2, since modern versions
# of numba don't support python 2
ENABLE_NUMBA = sys.version_info.major == 3


# whether to cache numba compilation (set to True if these function implementations will not change)
CACHE_NUMBA = False


# conditional numba import
if ENABLE_NUMBA:
    import numba


# numba decorator
def jit_decorator(func):
    if ENABLE_NUMBA:
        return numba.jit(nopython=True, cache=CACHE_NUMBA)(func)
    return func


@jit_decorator
def mat2quat(rmat, precise=False):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat: 3x3 rotation matrix
        precise: If isprecise is True, the input matrix is assumed to be a precise
             rotation matrix and a faster algorithm is used.

    Returns:
        vec4 float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]
    if precise:
        # This code uses a modification of the algorithm described in:
        # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
        # which is itself based on the method described here:
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        # Altered to work with the column vector convention instead of row vectors
        m = (
            M.conj().transpose()
        )  # This method assumes row-vector and postmultiplication of that vector
        if m[2, 2] < 0:
            if m[0, 0] > m[1, 1]:
                t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
                q = [m[1, 2] - m[2, 1], t, m[0, 1] + m[1, 0], m[2, 0] + m[0, 2]]
            else:
                t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
                q = [m[2, 0] - m[0, 2], m[0, 1] + m[1, 0], t, m[1, 2] + m[2, 1]]
        else:
            if m[0, 0] < -m[1, 1]:
                t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
                q = [m[0, 1] - m[1, 0], m[2, 0] + m[0, 2], m[1, 2] + m[2, 1], t]
            else:
                t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
                q = [t, m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0]]
        q = np.array(q)
        q *= 0.5 / np.sqrt(t)
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
                [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
                [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is Eigen vector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        inds = np.array([3, 0, 1, 2])
        q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


@jit_decorator
def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x, y, z, w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """

    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
    ...                                                direc, point)))
    True
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float64
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float64,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def make_pose(translation, rotation):
    """
    Make a homogenous pose matrix from a translation vector and a rotation matrix.

    :param translation: a 3-dim iterable
    :param rotation: a 3x3 matrix

    :return pose: a 4x4 homogenous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.
    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1.0]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def json_dump(dic, filename=None):
    """
    Dumps a python dictionary to a json file.
    If filename is not None, dump to file.
    Returns a string.
    """
    json_string = json.dumps(dic, indent=4, default=lambda o: "<not serializable>")
    if filename is not None:
        f = open(filename, "w")
        f.write(json_string)
        f.close()
    return json_string


def flatten_nested_dict(d, parent_key="", sep="_", item_key=""):
    """
    Flatten a nested dict to a list of key-value pairs. This function also works for hdf5 groups.
    Converting the output to a dictionary is easy as well - just call `dict` on it.
    """
    items = []
    if isinstance(d, dict) or isinstance(d, h5py.Group):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for k, v in d.items():
            k = str(k)
            assert isinstance(k, str)
            items.extend(flatten_nested_dict(v, new_key, sep=sep, item_key=k))
        return items
    else:
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        return [(new_key, d)]


class Rate(object):
    """
    Convenience class for enforcing rates in loops. Modeled after rospy.Rate.

    See http://docs.ros.org/en/jade/api/rospy/html/rospy.timer-pysrc.html#Rate.sleep
    """

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz

    def _remaining(self, curr_time):
        """
        Calculate time remaining for rate to sleep.
        """
        assert curr_time >= self.last_time, "time moved backwards!"
        elapsed = curr_time - self.last_time
        return self.sleep_duration - elapsed

    def sleep(self):
        """
        Attempt to sleep at the specified rate in hz, by taking the time
        elapsed since the last call to this function into account.
        """
        curr_time = time.time()
        remaining = self._remaining(curr_time)
        if remaining > 0:
            time.sleep(remaining)

        # assume successful rate sleeping
        self.last_time = self.last_time + self.sleep_duration

        # NOTE: this commented line is what we used to do, but this enforces a slower rate
        # self.last_time = time.time()

        # detect time jumping forwards (e.g. loop is too slow)
        if curr_time - self.last_time > self.sleep_duration * 2:
            # we didn't sleep at all
            self.last_time = curr_time


class FunctionTimer:
    """
    A utility class to measure and track the execution time of functions.

    This class provides both a context manager and a decorator for timing functions.
    It can track multiple function calls and provide statistics on execution times.

    Attributes:
        history (Dict[str, List[float]]): A dictionary mapping function names to lists
            of execution times in seconds.
        last_result (Dict[str, float]): A dictionary mapping function names to their
            most recent execution time in seconds.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.last_result: Dict[str, float] = {}

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to time a function.

        Args:
            func: The function to be timed.

        Returns:
            A wrapped function that will be timed when called.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.measure(func.__name__):
                return func(*args, **kwargs)

        return wrapper

    def measure(self, name: Optional[str] = None) -> "TimerContext":
        """
        Context manager for timing a block of code.

        Args:
            name: An optional name for the timed block. If None, a default name will be used.

        Returns:
            A TimerContext object to be used in a with statement.
        """
        return TimerContext(self, name)

    def record_time(self, name: str, elapsed_time: float) -> None:
        """
        Record the execution time for a named function or block.

        Args:
            name: The name of the function or block.
            elapsed_time: The execution time in seconds.
        """
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(elapsed_time)
        self.last_result[name] = elapsed_time

    def get_last_time(self, name: str) -> Optional[float]:
        """
        Get the most recent execution time for a named function or block.

        Args:
            name: The name of the function or block.

        Returns:
            The most recent execution time in seconds, or None if the function
            has not been timed yet.
        """
        return self.last_result.get(name)

    def get_average_time(self, name: str) -> Optional[float]:
        """
        Get the average execution time for a named function or block.

        Args:
            name: The name of the function or block.

        Returns:
            The average execution time in seconds, or None if the function
            has not been timed yet.
        """
        times = self.history.get(name)
        if times:
            return sum(times) / len(times)
        return None

    def get_stats(self, name: str) -> Dict[str, Union[float, int, None]]:
        """
        Get comprehensive statistics for a named function or block.

        Args:
            name: The name of the function or block.

        Returns:
            A dictionary containing statistics such as min, max, avg, count, and total.
        """
        times = self.history.get(name)
        if not times:
            return {"min": None, "max": None, "avg": None, "count": 0, "total": 0}

        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times[-100:]) / min(len(times), 100),
            "count": len(times),
            "total": sum(times),
            "median": np.median(times),
        }

    def clear(self, name: Optional[str] = None) -> None:
        """
        Clear the timing history.

        Args:
            name: If provided, only clear the history for this name.
                 If None, clear all history.
        """
        if name is None:
            self.history.clear()
            self.last_result.clear()
        else:
            if name in self.history:
                del self.history[name]
            if name in self.last_result:
                del self.last_result[name]

    def format_time(self, seconds: float) -> str:
        """
        Format a time in seconds to a human-readable string.

        Args:
            seconds: The time in seconds.

        Returns:
            A formatted string representation of the time.
        """
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.2f} μs"
        elif seconds < 1:
            return f"{seconds * 1_000:.2f} ms"
        else:
            return str(timedelta(seconds=seconds))

    def print_stats(self, name: Optional[str] = None) -> None:
        """
        Print statistics for a function or all functions.

        Args:
            name: If provided, only print stats for this name.
                 If None, print stats for all functions.
        """
        if name is not None:
            stats = self.get_stats(name)
            if stats["count"] > 0:
                print(f"Stats for {name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Total: {self.format_time(stats['total'])}")
                print(f"  Average: {self.format_time(stats['avg'])}")
                print(f"  Min: {self.format_time(stats['min'])}")
                print(f"  Max: {self.format_time(stats['max'])}")
                print(f"  Median: {self.format_time(stats['median'])}")
            else:
                print(f"No stats available for {name}")
        else:
            if not self.history:
                print("No timing data available")
                return

            print("Function Timing Statistics:")
            for func_name in sorted(self.history.keys()):
                stats = self.get_stats(func_name)
                print(f"\n{func_name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Total: {self.format_time(stats['total'])}")
                print(f"  Average: {self.format_time(stats['avg'])}")
                print(f"  Min: {self.format_time(stats['min'])}")
                print(f"  Max: {self.format_time(stats['max'])}")
                print(f"  Median: {self.format_time(stats['median'])}")


class TimerContext:
    """
    A context manager for timing blocks of code.

    This class is used internally by FunctionTimer and should not be
    instantiated directly.
    """

    def __init__(self, timer: FunctionTimer, name: Optional[str] = None):
        """
        Initialize a TimerContext.

        Args:
            timer: The FunctionTimer instance.
            name: An optional name for the timed block. If None, a default name will be used.
        """
        self.timer = timer
        self.name = name or "unnamed_block"
        self.start_time = 0

    def __enter__(self) -> "TimerContext":
        """Start the timer when entering the context."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Stop the timer when exiting the context and record the elapsed time.

        Args:
            exc_type: The exception type if an exception was raised.
            exc_val: The exception value if an exception was raised.
            exc_tb: The traceback if an exception was raised.
        """
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time
        self.timer.record_time(self.name, elapsed_time)


def get_utilization_percentages(reset: bool = False, max_values: List[float] = [0.0, 0.0, 0.0, 0.0]) -> List[float]:
    """Get the maximum CPU, RAM, GPU utilization (processing), and
    GPU memory usage percentages since the last time reset was true."""

    if reset:
        max_values[:] = [0, 0, 0, 0]  # Reset the max values

    # CPU utilization
    cpu_usage = psutil.cpu_percent(interval=0.1)
    max_values[0] = max(max_values[0], cpu_usage)

    # RAM utilization
    memory_info = psutil.virtual_memory()
    ram_usage = memory_info.percent
    max_values[1] = max(max_values[1], ram_usage)

    # GPU utilization using pynvml
    if torch.cuda.is_available():
        pynvml.nvmlInit()  # Initialize NVML
        for i in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU Utilization
            gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_processing_utilization_percent = gpu_utilization.gpu  # GPU core utilization
            max_values[2] = max(max_values[2], gpu_processing_utilization_percent)

            # GPU Memory Usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_total = memory_info.total
            gpu_memory_used = memory_info.used
            gpu_memory_utilization_percent = (gpu_memory_used / gpu_memory_total) * 100
            max_values[3] = max(max_values[3], gpu_memory_utilization_percent)

        pynvml.nvmlShutdown()  # Shutdown NVML after usage
    else:
        gpu_processing_utilization_percent = None
        gpu_memory_utilization_percent = None
    return max_values
