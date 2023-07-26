#!/usr/bin/env python3

import numpy as np
import argparse
import math
import os

TIMESTAMPS_FILENAME = "timestamps.txt"
OXTS_DIRNAME = "oxts"
DATA_DIRNAME = "data"
OXTS_LINE_LEN = 30


def load_timestamps(ts_dir):
    ts_file = os.path.join(ts_dir, TIMESTAMPS_FILENAME)
    # TODO check float precision
    return np.loadtxt(ts_file, delimiter=",", converters={0: lambda v: np.int64(np.datetime64(v)) / 1e+9 })


def load_oxts_data(base_dir):
    oxts_dir = os.path.join(base_dir, OXTS_DIRNAME)
    # FIXME the content of ts is useless here
    ts = load_timestamps(oxts_dir)
    ts_len = len(ts)
    oxts_data = np.zeros((ts_len, OXTS_LINE_LEN))
    for i in range(ts_len):
        data_dir = os.path.join(oxts_dir, DATA_DIRNAME)
        t_filename = str(i).zfill(10) + ".txt"
        oxts_data[i] = np.loadtxt(os.path.join(data_dir, t_filename))
    return oxts_data


def lat_lon_to_mercator(lat, lon, scale):
    """
    Converts lat/lon coordinates to mercator coordinates using Mercator scale.
    """
    ER = 6378137
    mx = scale * lon * math.pi * ER / 180
    my = scale * ER * math.log(math.tan((90 + lat) * math.pi / 360))
    return mx, my


def lat_to_scale(lat):
    """
    Compute Mercator scale from latitude
    """
    return math.cos(lat * math.pi / 180.0)


def gps_imu_to_pose(gps_imu, scale):
    """
    Compute pose from GPS/IMU data
    """
    t = np.zeros(3)
    t[0], t[1] = lat_lon_to_mercator(gps_imu[0], gps_imu[1], scale)
    t[2] = gps_imu[2]  # altitude
    rx = gps_imu[3]  # roll
    ry = gps_imu[4]  # pitch
    rz = gps_imu[5]  # heading
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def oxts_to_pose(oxts_info):
    """
    Converts a list of oxts measurements into metric poses,
    starting at (0,0,0) meters, OXTS coordinates are defined as
    x = forward, y = right, z = down (see OXTS RT3000 user manual)
    afterwards, pose{i} contains the transformation which takes a
    3D point in the i'th frame and projects it into the oxts
    coordinates of the first frame.
    """
    # Compute scale from first lat value
    scale = lat_to_scale(oxts_info[0, 0])
    Tr_0_inv = None
    poses = np.zeros((len(oxts_info), 12))
    for i, line in enumerate(oxts_info):
        T = gps_imu_to_pose(line, scale)
        # Normalize translation and rotation (start at 0/0/0)
        if Tr_0_inv is None:
            Tr_0_inv = np.linalg.inv(T)

        pose = Tr_0_inv.dot(T)
        poses[i] = pose[:3, :].reshape(12)
    return poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert KITTI GPS/IMU data to pose format.")
    parser.add_argument("basedir", help="Path to base dir.")
    parser.add_argument("--output", default="poses.txt", help="Path to output file")
    args = parser.parse_args()
    oxts_data = load_oxts_data(args.basedir)
    poses = oxts_to_pose(oxts_data)
    np.savetxt(args.basedir + 'poses.txt', poses)