# Copyright (c) 2022 Carnegie Mellon University <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)

"""
optitrack utilities 
"""

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import pandas as pd


def parse_optitrack_csv(csvFile: str, bodies: list, markers: bool = False):
    # bodies = bodies[0].split()
    print("\n--------------------------------")
    print(f"CSV file: {csvFile}")
    print(f"Bodies to track: {bodies}")
    print("--------------------------------")

    meta = metadata(csvFile)
    print(meta)

    body_poses = {}
    for body in bodies:
        body_poses[body] = rigid_body(body, csvFile, markers=markers)

    return {"metadata": meta, "poses": body_poses}


def clean_up_optitrack(poses):
    """
    Filter large jumps in mocap data
    """
    traj_sz = poses.shape[0]
    diff_pose_mags = []
    adjusted_count = 0
    filtered_poses = np.empty((0, 4, 4))
    for i in range(traj_sz):
        if i > 0:
            diff_pose = np.linalg.inv(poses[i - 1, :]) @ poses[i, :]
            diff_pose_mag = np.linalg.norm(diff_pose[:3, 3])
            diff_pose_mags.append(diff_pose_mag)
            avg_diff_pose_mag = sum(diff_pose_mags) / len(diff_pose_mags)
            if i > 1 and diff_pose_mag > 10 * avg_diff_pose_mag:
                adjusted_count += 1
            else:
                filtered_poses = np.concatenate(
                    (filtered_poses, poses[i, :][None, :]), axis=0
                )
                # print(f"Jump @ t = {i} : avg: {avg_diff_pose_mag}, curr: {diff_pose_mag}")
                # poses[i, :] = tf_to_xyzquat(xyzquat_to_tf(poses[i - 1, :]) @ xyzquat_to_tf(prev_diff_pose))
            # prev_diff_pose = diff_pose
    print(f"Adjusted {adjusted_count} / {traj_sz} object-sensor poses")
    return filtered_poses


def pose_to_axes(quaternions: np.ndarray):
    """
    Convert quaternion to viz pose
    """
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    r = R.from_quat(quaternions)  # (N, 4) [qx, qy, qz, qw]
    quivers_u = r.apply(x)
    quivers_v = r.apply(y)
    quivers_w = r.apply(z)
    return quivers_u, quivers_v, quivers_w


def plot_optitrack_poses(body_poses: dict, save_path: str):
    """
    plot the rigid body frames for the full trajectory
    """

    bodies = list(body_poses.keys())
    colors = cm.Set1(np.linspace(0, 1, len(bodies)))
    print("Plotting bodies: {}".format(bodies))
    fig = plt.figure()

    ax = plt.subplot(1, 1, 1, projection="3d")
    axes_sz = 2e-2
    for i, body in enumerate(bodies):

        if type(body_poses[body]) is np.ndarray:
            x, y, z = (
                body_poses[body][:, 0],
                body_poses[body][:, 1],
                body_poses[body][:, 2],
            )
            ax.plot(x, y, z, linestyle="-", label=body, color=colors[i], markersize=1)
            u, v, w = pose_to_axes(body_poses[body][:, 3:])
            ax.quiver(
                x,
                y,
                z,
                u[:, 0],
                u[:, 1],
                u[:, 2],
                length=axes_sz,
                color="r",
                linewidths=0.5,
                alpha=0.5,
                normalize=True,
            )
            ax.quiver(
                x,
                y,
                z,
                v[:, 0],
                v[:, 1],
                v[:, 2],
                length=axes_sz,
                color="g",
                linewidths=0.5,
                alpha=0.5,
                normalize=True,
            )
            ax.quiver(
                x,
                y,
                z,
                w[:, 0],
                w[:, 1],
                w[:, 2],
                length=axes_sz,
                color="b",
                linewidths=0.5,
                alpha=0.5,
                normalize=True,
            )
        else:
            x, y, z = (
                body_poses[body].position[:, 0],
                body_poses[body].position[:, 1],
                body_poses[body].position[:, 2],
            )
            ax.plot(x, y, z, linestyle="-", label=body, color=colors[i], markersize=1)
            u, v, w = pose_to_axes(body_poses[body].quaternion)
            ax.quiver(
                x,
                y,
                z,
                u[:, 0],
                u[:, 1],
                u[:, 2],
                length=axes_sz,
                color="r",
                linewidths=0.5,
                alpha=0.5,
                normalize=True,
            )
            ax.quiver(
                x,
                y,
                z,
                v[:, 0],
                v[:, 1],
                v[:, 2],
                length=axes_sz,
                color="g",
                linewidths=0.5,
                alpha=0.5,
                normalize=True,
            )
            ax.quiver(
                x,
                y,
                z,
                w[:, 0],
                w[:, 1],
                w[:, 2],
                length=axes_sz,
                color="b",
                linewidths=0.5,
                alpha=0.5,
                normalize=True,
            )

    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    print(
        "Saving rigid body plot: {}".format(
            os.path.join(save_path, "3d_rigid_bodies.pdf")
        )
    )
    plt.savefig(
        os.path.join(save_path, "3d_rigid_bodies.pdf"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_optitrack_markers(body_poses: dict, save_path: str):
    """
    plot all individual markers for the full trajectory
    """

    bodies = list(body_poses.keys())
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1, projection="3d")
    for body in bodies:
        num_markers = len(body_poses[body].markers)
        print("Plotting {} {} markers".format(num_markers, body))
        colors = cm.Set1(np.linspace(0, 1, num_markers))
        for j in range(1, 1 + num_markers):
            x, y, z = (
                body_poses[body].markers[j][:, 0],
                body_poses[body].markers[j][:, 1],
                body_poses[body].markers[j][:, 2],
            )
            ax.plot(
                x,
                y,
                z,
                linestyle="-",
                color=colors[j - 1],
                markersize=0.5,
                linewidth=0.5,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    print("Saving marker plot: {}".format(os.path.join(save_path, "3d_markers.pdf")))
    plt.savefig(
        os.path.join(save_path, "3d_markers.pdf"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_marker_dist(body_poses: dict, save_path: str, has_gt: bool = False):
    """
    Plot the average euclidean distance between all the markers in a body
    """
    bodies = list(body_poses.keys())
    from scipy.spatial import distance_matrix

    # GT distances of optitrack CS-200 square (https://d111srqycjesc9.cloudfront.net/CS-200TechnicalDrawing.pdf)
    meanD_gt = np.mean([0.150, 0.200, 0.250]) if has_gt else 0.0

    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    colors = cm.tab10(np.linspace(0, 1, len(bodies)))

    for i, body in enumerate(bodies):
        num_markers = len(body_poses[body].markers)
        T = body_poses[body].markers[1].shape[0]
        times = body_poses[body].time
        errors = []
        # loop over timesteps
        for t in range(T):
            markers = []
            for j in range(1, 1 + num_markers):
                markers.append(body_poses[body].markers[j][t, :])

            # mean constellation distance
            D = distance_matrix(markers, markers)
            D = D.flatten()
            D = D[D != 0]
            D = np.unique(D)
            meanD = np.mean(D)
            error = np.abs(meanD_gt - meanD) * 1e3
            errors.append(error)

        ax.plot(
            times, errors, linestyle="-", color=colors[i], markersize=0.5, linewidth=1
        )

    ax.set_xlabel("timesteps")
    if has_gt:
        ax.set_ylabel("avg. constellation error (mm)")
    else:
        ax.set_ylabel("avg. constellation disatance (mm)")
    fig.tight_layout()
    print(
        "Saving constellation plot: {}".format(
            os.path.join(save_path, "constellation_dist.pdf")
        )
    )
    plt.savefig(
        os.path.join(save_path, "constellation_dist.pdf"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_optitrack_derivatives(body_poses: dict, save_path: str):
    """
    Plotting the velocity and acceleration of the rigid body frames
    """

    bodies = list(body_poses.keys())
    fig = plt.figure()
    ax = plt.subplot(2, 1, 1)
    for body in bodies:
        times = body_poses[body].time
        positions = body_poses[body].position
        velocities = np.zeros_like(positions)
        for j in range(1, times.shape[0]):
            velocities[j, :] = (positions[j, :] - positions[j - 1, :]) / (
                times[j] - times[j - 1]
            )

        accelerations = np.zeros_like(velocities)
        for j in range(1, times.shape[0]):
            accelerations[j, :] = (velocities[j, :] - velocities[j - 1, :]) / (
                times[j] - times[j - 1]
            )

        # vx = ax.plot(times, velocities[:, 0], linestyle = '-', markersize = 0.5, label = 'Xvel', linewidth = 1)
        # vy = ax.plot(times, velocities[:, 1], linestyle = '-', markersize = 0.5, label = 'Yvel', linewidth = 1)
        # vz = ax.plot(times, velocities[:, 2], linestyle = '-', markersize = 0.5, label = 'Zvel', linewidth = 1)
        vel_mag = np.linalg.norm(velocities, axis=1)
        ax.plot(times, vel_mag, linestyle="-", markersize=0.5, color="k", linewidth=1)

        ax.set_xlabel("Time (s)")
        # ax.set_ylabel('Rigid body vel. (m/s)')
        ax.set_ylabel("vel. magnitude (m/s)")

        ax = plt.subplot(2, 1, 2)
        # ax.plot(times, accelerations[:, 0], linestyle = '--', markersize = 0.5, color = vx[0].get_c(), label = 'Xvel', linewidth = 1)
        # ax.plot(times, accelerations[:, 1], linestyle = '--', markersize = 0.5, color = vy[0].get_c(),label = 'Yvel', linewidth = 1)
        # ax.plot(times, accelerations[:, 2], linestyle = '--', markersize = 0.5, color = vz[0].get_c(),label = 'Zvel', linewidth = 1)
        acc_mag = np.linalg.norm(accelerations, axis=1)
        ax.plot(times, acc_mag, linestyle="--", markersize=0.5, color="k", linewidth=1)

        ax.set_xlabel("Time (s)")
        # ax.set_ylabel('Rigid body acc. (m/s^2)')
        ax.set_ylabel("acc. magnitude (m/s^2)")

    fig.tight_layout()
    print(
        "Saving derivatives plot: {}".format(os.path.join(save_path, "derivatives.pdf"))
    )
    plt.savefig(
        os.path.join(save_path, "derivatives.pdf"),
        transparent=True,
        bbox_inches="tight",
        pad_inches=0,
    )


# Modified from original source: https://github.com/timeanddoctor/parse_optitrack_csv/tree/master

###########################################################################
#####                        parseOptitrackCSV                        #####
#####                            Raul Tapia                           #####
#####      GRVC Robotics Laboratory at the University of Seville      #####
###########################################################################

# @file    parseOptitrackCSV.py
# @brief   Parse exported csv files from Motive software.
# @author  Raul Tapia


class marker:
    """
    Single marker class
    """

    position = []

    # Constructor for marker
    # @param   px   Position of the marker in x axis
    # @param   py   Position of the marker in y axis
    # @param   pz   Position of the marker in z axis
    # @param   q    Quality of the marker
    def __init__(self, px, py, pz, q):
        position = (px, py, pz)


class metadata:
    """
    Experiment data
    """

    expt_name = None
    frame_rate = None
    start_time = None
    length = None

    def __init__(self, filename):
        metadata_row = (
            pd.read_csv(filename, nrows=1, header=None).to_numpy().squeeze().tolist()
        )
        self.expt_name = metadata_row[3]
        self.frame_rate = metadata_row[7]
        self.start_time = metadata_row[11]
        self.length = metadata_row[15]

    def __str__(self):
        return "Name: {}, Frame rate: {}, Start time: {}, Length: {}".format(
            self.expt_name, self.frame_rate, self.start_time, self.length
        )

    def __len__(self):
        return self.length


class rigid_body:
    """
    Tracking pose data
    """

    frame = []
    time = []
    quaternion = []
    position = []
    markers = {}
    framesWithError = []
    name = None

    # Constructor for rigid_body
    # @param   filename   Name of the csv file
    def __init__(self, name, filename, markers=False):
        ### Load data
        self.name = name
        (
            self.frame,
            self.time,
            self.quaternion,
            self.position,
            self.marker,
            self.framesWithError,
        ) = ([], [], [], [], [], [])

        print("Parsing {}".format(self.name))

        types = (
            pd.read_csv(filename, skiprows=2, nrows=1, header=None)
            .to_numpy()
            .squeeze()
            .tolist()
        )
        bodies = (
            pd.read_csv(filename, skiprows=3, nrows=1, header=None)
            .to_numpy()
            .squeeze()
            .tolist()
        )
        bodies = [str(x) for x in bodies]

        meas_cols = [
            "Frame",
            "Time",
            "rotationX",
            "rotationY",
            "rotationZ",
            "rotationW",
            "positionX",
            "positionY",
            "positionZ",
        ]
        use_cols = [0, 1]  # frame, time
        for i, b in enumerate(bodies):
            if b == self.name:
                use_cols.append(i)

        rigid_body_markers = [i for i, e in enumerate(types) if e == "marker"]

        marker_idxs = {}
        for i in range(1, 5):
            m = list(
                filter(
                    lambda x: x[1].startswith(self.name + ":marker" + str(i)),
                    enumerate(bodies),
                )
            )
            m = [x[0] for x in m]

            m = list(set(m) & set(rigid_body_markers))
            if m:
                marker_idxs[i] = m
                self.markers[i] = []
            else:
                break

        # switch to loading pandas dataframe here
        reader = pd.read_csv(
            filename, skiprows=7, usecols=use_cols, names=meas_cols, header=None
        )

        ### Instance reader
        N = reader.shape[0]
        for i in tqdm(range(N)):
            row = reader.iloc[i]
            self.frame.append(int(row["Frame"]))
            self.time.append(float(row["Time"]))

            ### Check if error
            if row["rotationX"] == "":
                self.quaternion.append("?")
                self.position.append("?")
                self.marker.append("?")
                self.framesWithError.append(self.frame[-1])
            else:
                self.quaternion.append(
                    (
                        float(row["rotationX"]),
                        float(row["rotationY"]),
                        float(row["rotationZ"]),
                        float(row["rotationW"]),
                    )
                )
                self.position.append(
                    (
                        float(row["positionX"]),
                        float(row["positionY"]),
                        float(row["positionZ"]),
                    )
                )
        self.frame = np.stack(self.frame, axis=0)
        self.time = np.stack(self.time, axis=0)
        self.quaternion = np.stack(self.quaternion, axis=0)
        self.position = np.stack(self.position, axis=0)

        ## markers
        if markers:
            meas_cols = ["positionX", "positionY", "positionZ"]
            for key, val in marker_idxs.items():
                reader = pd.read_csv(
                    filename, skiprows=7, usecols=val, names=meas_cols, header=None
                )
                N = reader.shape[0]
                for i in range(N):
                    row = reader.iloc[i]
                    if row["positionX"] == "":
                        continue
                    else:
                        self.markers[key].append(
                            (
                                float(row["positionX"]),
                                float(row["positionY"]),
                                float(row["positionZ"]),
                            )
                        )
                self.markers[key] = np.stack(self.markers[key], axis=0)

    def __len__(self):
        return len(self.position)
