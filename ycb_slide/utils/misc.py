# Copyright (c) 2022 Carnegie Mellon University <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)

from datetime import datetime
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial.transform import Rotation as R
import dill as pickle
from .optitrack import clean_up_optitrack
import matplotlib.pyplot as plt


def tf2positionquat(pose):
    pose = np.atleast_3d(pose)
    pose = np.rollaxis(pose, 2)  # (4, 4, N) --> (N, 4, 4)
    # convert 4 x 4 transformation matrix to [x, y, z, qx, qy, qz, qw]
    r = R.from_matrix(np.array(pose[:, 0:3, 0:3]))
    q = r.as_quat()  # qx, qy, qz, qw
    t = pose[:, 0:3, 3]
    return np.concatenate((t, q), axis=1)  # (N, 7)


def positionquat2tf(position_quat):
    try:
        position_quat = np.atleast_2d(position_quat)
        # position_quat : N x 7
        N = position_quat.shape[0]
        T = np.zeros((4, 4, N))
        T[0:3, 0:3, :] = np.moveaxis(
            R.from_quat(position_quat[:, 3:]).as_matrix(), 0, -1
        )
        T[0:3, 3, :] = position_quat[:, :3].T
        T[3, 3, :] = 1
    except ValueError:
        print("Zero quat error!")
    return T.squeeze() if N == 1 else T


def find_nearest(array, value):
    """
    return index and value of nearest value in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def current_time():
    """
    Get current time
    """
    return datetime.now()


def datetime_to_str(time_obj):
    """
    Convert time object to full string e.g. 2022-03-20 06:40:57.408349 PM
    """
    return time_obj.strftime("%Y-%m-%d %I:%M:%S:%f %p")


def str_to_datetime(time_string):
    """
    Convert time string back to datetime object
    """
    time_string = time_string.replace(".", ":")
    return datetime.strptime(time_string, "%Y-%m-%d %I:%M:%S:%f %p")


def datetime_diff(curr_time_obj, ref_time_obj):
    """
    Difference between two datetime objects in seconds
    """
    return (curr_time_obj - ref_time_obj).total_seconds()


def utc_to_local(
    time_string: str, return_string: bool = False, time_zone: str = "America/New_York"
):
    """
    Convert UTC time to local timezone
    """
    from dateutil import tz

    from_zone = tz.gettz("UTC")
    to_zone = tz.gettz(time_zone)
    utc = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S.%f")
    utc = utc.replace(tzinfo=from_zone)

    if return_string:
        return utc.astimezone(to_zone).strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        return utc.astimezone(to_zone)


def sync_optitrack_digit(tracking_data: dict, digit_data: dict):
    # get the time offset: tracking start time is always ahead of experiment start time
    tracking_start_time = str_to_datetime(tracking_data["metadata"].start_time)
    experiment_start_time = str_to_datetime(digit_data["start_time"])

    print(
        "Tracking start time: {} Expt start time: {}".format(
            tracking_data["metadata"].start_time, digit_data["start_time"]
        )
    )
    diff_time = datetime_diff(experiment_start_time, tracking_start_time)

    if diff_time < 0 or diff_time > 20 * 60:
        print("tracking and expriement mismatch!")
        return None

    bodies = list(tracking_data["poses"].keys())
    # get the object poses
    experiment_length = len(digit_data["timestamps"])
    poses = {k: [None] * experiment_length for k in bodies}

    for i in range(experiment_length):
        t_experiment = digit_data["timestamps"][i] + diff_time
        for body in bodies:
            val, idx = find_nearest(tracking_data["poses"][body].time, t_experiment)
            # print("Diff: {}".format(val - t_experiment))
            nearest_pose = np.hstack(
                (
                    tracking_data["poses"][body].position[idx, :],
                    tracking_data["poses"][body].quaternion[idx, :],
                )
            )  # [x, y, z, qx, qy, qz, qw]
            poses[body][i] = nearest_pose

    for body in bodies:
        poses[body] = np.vstack(poses[body])
    digit_data["poses"] = poses
    return digit_data


def pick_points(mesh_path):
    """
    http://www.open3d.org/docs/latest/tutorial/visualization/interactive_visualization.html
    """
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return np.asarray(pcd.points)[vis.get_picked_points(), :]


def pose2quiver(poses, sz):
    """
    Convert pose to quiver object (RGB)
    """
    poses = np.atleast_3d(poses)
    quivers = pv.PolyData(poses[:, :3, 3])  # (N, 3) [x, y, z]
    x, y, z = np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])
    r = R.from_matrix(poses[:, 0:3, 0:3])
    quivers["xvectors"], quivers["yvectors"], quivers["zvectors"] = (
        r.apply(x) * sz,
        r.apply(y) * sz,
        r.apply(z) * sz,
    )
    return quivers


def draw_poses(
    plotter: pv.Plotter,
    mesh: pv.DataSet,
    cluster_poses: np.ndarray,
    opacity: float = 1.0,
) -> None:
    """
    Draw pose RGB coordinate axes for pose set in pyvista visualizer
    """
    quivers = pose2quiver(cluster_poses, mesh.length / 20)
    quivers = [quivers["xvectors"]] + [quivers["yvectors"]] + [quivers["zvectors"]]
    names = ["xvectors", "yvectors", "zvectors"]
    colors = ["r", "g", "b"]
    cluster_centers = cluster_poses[:, :3, 3]
    for (q, c, n) in zip(quivers, colors, names):
        plotter.add_arrows(
            cluster_centers,
            q,
            color=c,
            opacity=opacity,
            show_scalar_bar=False,
            render=False,
            name=n,
        )


def viz_poses_pointclouds_on_mesh(
    mesh_path, poses, pointclouds, save_path=None, decimation_factor=5
):
    """
    Visualize poses on the ground-truth mesh model
    """
    if type(pointclouds) is not list:
        temp = pointclouds
        pointclouds = [None] * 1
        pointclouds[0] = temp

    plotter = pv.Plotter(window_size=[2000, 2000])

    mesh = pv.read(mesh_path)  # pyvista object
    dargs = dict(
        color="grey",
        ambient=0.6,
        opacity=0.5,
        smooth_shading=True,
        specular=1.0,
        show_scalar_bar=False,
        render=False,
    )
    plotter.add_mesh(mesh, **dargs)
    draw_poses(plotter, mesh, poses)

    if poses.ndim == 2:
        spline = pv.lines_from_points(poses[:, :3])
        plotter.add_mesh(spline, line_width=3, color="k")

    final_pc = np.empty((0, 3))
    for i, pointcloud in enumerate(pointclouds):
        if pointcloud.shape[0] == 0:
            continue
        if decimation_factor is not None:
            downpcd = pointcloud[
                np.random.choice(
                    pointcloud.shape[0],
                    pointcloud.shape[0] // decimation_factor,
                    replace=False,
                ),
                :,
            ]
        else:
            downpcd = pointcloud
        final_pc = np.append(final_pc, downpcd)

    if final_pc.shape[0]:
        pc = pv.PolyData(final_pc)
        plotter.add_points(
            pc, render_points_as_spheres=True, color="#26D701", point_size=3
        )

    if save_path:
        plotter.show(screenshot=save_path)
        print(f"Save path: {save_path}.png")
    else:
        plotter.show()
    plotter.close()
    pv.close_all()


def cam2gel(cam_pose, cam_dist):
    """
    Convert cam_pose to gel_pose
    """
    cam_tf = np.eye(4)
    cam_tf[2, 3] = -cam_dist
    return cam_pose @ cam_tf[None, :]


def process_sim_poses(pickle_file):
    """
    Convert raw sim data to (N, 4, 4) transformations in object-centric frame
    """
    with open(pickle_file, "rb") as p:
        poses = pickle.load(p)
    camposes, gelposes, gelposes_meas = (
        poses["camposes"],
        poses["gelposes"],
        poses["gelposes_meas"],
    )
    digit_poses = positionquat2tf(gelposes_meas)
    digit_poses = np.rollaxis(digit_poses, 2)
    return digit_poses


def process_real_poses(npy_file, alignment_file, object_name):
    """
    Convert raw real data to (N, 4, 4) transformations in object-centric frame
    """
    digit_data = np.load(npy_file, allow_pickle=True).item()

    digit_poses = digit_data["poses"]["DIGIT"]
    obj_poses = digit_data["poses"][object_name]
    digit_poses, obj_poses = positionquat2tf(digit_poses), positionquat2tf(obj_poses)
    digit_poses, obj_poses = np.rollaxis(digit_poses, 2), np.rollaxis(
        obj_poses, 2
    )  # (4, 4, N) --> (N, 4, 4)

    digit_poses = np.linalg.inv(obj_poses) @ digit_poses  # relative to object
    digit_poses = clean_up_optitrack(digit_poses)  # remove jumps
    digit_poses = cam2gel(digit_poses, cam_dist=0.022)  # convert to contact poses

    # align manually
    alignment = np.load(alignment_file)

    # Adjust only translation of poses
    pose = np.eye(4)
    pose = np.repeat(pose[None, :], digit_poses.shape[0], axis=0)
    pose[:, :3, 3] = digit_poses[:, :3, 3]
    pose = pose @ alignment[None, :]
    digit_poses[:, :3, 3] = pose[:, :3, 3]
    return digit_poses
