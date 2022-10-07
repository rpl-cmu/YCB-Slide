
# Copyright (c) 2022 Carnegie Mellon University <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)

"""
Loads all real datasets for object and manually corrects object pose (alignment.npy)
python align_data.py --data_path dataset/real/035_power_drill --object 035_power_drill
"""

import os
from os import path as osp
import numpy as np
from utils.misc import cam2gel, viz_poses_pointclouds_on_mesh, positionquat2tf
from utils.optitrack import clean_up_optitrack
import argparse
import open3d as o3d

def align_data(source_cloud, target_mesh, N=5000):
    source_cloud = np.vstack(source_cloud)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(source_cloud)

    obj_mesh = o3d.io.read_triangle_mesh(target_mesh)
    obj_cloud = obj_mesh.sample_points_uniformly(number_of_points=N)

    print(f"Tactile points: {len(pcd.points)}, Mesh points: {len(obj_cloud.points)}")
    trans_init = np.eye(4)

    source, target = pcd, obj_cloud
    # pick points from two point clouds and builds correspondences
    # first point : tactile point clouds
    # second point : obj cloud
    picked_id = pick_alignment_points(source, target)

    if not len(picked_id):
        return np.eye(4)

    assert len(picked_id) >= 3 * 2 and len(picked_id) % 2 == 0
    corr = np.zeros((len(picked_id) // 2, 2))
    corr[:, 0] = picked_id[0::2]
    corr[:, 1] = picked_id[1::2]
    corr[:, 1] = [x - len(source.points) for x in corr[:, 1]]

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(
        source, target, o3d.utility.Vector2iVector(corr)
    )

    return trans_init

def pick_alignment_points(source, target):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([source.points, target.points]))
    c1 = np.repeat(np.array([[0, 0, 1]]), repeats=len(source.points), axis=0)
    c2 = np.repeat(np.array([[0, 1, 0]]), repeats=len(target.points), axis=0)
    c = np.vstack([c1, c2])
    pcd.colors = o3d.utility.Vector3dVector(c)

    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def align_data(data_path, object_name):
    dname = osp.dirname(osp.abspath(__file__))
    os.chdir(dname)
    os.chdir("..")  # root

    obj_path = osp.join("dataset", "obj_models", object_name, "nontextured.stl")

    all_datasets = sorted(os.listdir(data_path))

    all_pointclouds_w = []
    all_digit_poses =  np.empty((0, 4, 4))
    for dataset in all_datasets:
        if dataset == "bg" or not osp.isdir(osp.join(data_path, dataset)):
            continue
        dataset_path = osp.join(data_path, dataset)

        # make paths
        pose_file = osp.join(dataset_path, "synced_data.npy")
        print(f"Object: {object_name}, Dataset: {dataset}\n")

        print("Loading data: {}".format(pose_file))
        digit_data = np.load(pose_file, allow_pickle=True).item()

        digit_poses = digit_data["poses"]["DIGIT"]
        obj_poses = digit_data["poses"][object_name]
        digit_poses, obj_poses = positionquat2tf(digit_poses), positionquat2tf(obj_poses)
        digit_poses, obj_poses = np.rollaxis(digit_poses,2), np.rollaxis(obj_poses,2) # (4, 4, N) --> (N, 4, 4)

        digit_poses = np.linalg.inv(obj_poses) @ digit_poses  # relative to object
        digit_poses = clean_up_optitrack(digit_poses) # remove jumps
        digit_poses = cam2gel(digit_poses, cam_dist = 0.022) # convert to contact poses

        pointclouds_w = digit_poses[:, :3, 3]
        all_pointclouds_w += [pointclouds_w]
        all_digit_poses = np.concatenate((all_digit_poses, digit_poses), axis=0)

    # align manually
    alignment_file = osp.join(data_path, "alignment.npy")
    if not osp.exists(alignment_file):
        alignment = align_data(
            source_cloud=all_pointclouds_w, target_mesh=obj_path
        )
        np.save(alignment_file, alignment)
    else:
        reply = (
            str(
                input("Alignment file already exists, manual align again?" + " (y/n): ")
            )
            .lower()
            .strip()
        )
        if reply[0] == "y":
            alignment = align_data(
                source_cloud=all_pointclouds_w, target_mesh=obj_path
            )
            np.save(alignment_file, alignment)
        else:
            alignment = np.load(alignment_file)

    # Adjust only translation of poses 
    pose = np.eye(4)
    pose = np.repeat(pose[None, :], all_digit_poses.shape[0], axis=0)
    pose[:, :3, 3] = all_digit_poses[:, :3, 3]
    pose  = pose @ alignment[None, :] 
    all_digit_poses[:, :3, 3] = pose[:, :3, 3]

    print("Visualizing alignment results")
    viz_poses_pointclouds_on_mesh(
        mesh_path=obj_path,
        poses=all_digit_poses,
        pointclouds=all_pointclouds_w,
        save_path=osp.join(data_path, "tactile_data"),
        decimation_factor=None,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align synced optitrack + digit data")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to synced tactile data + optitrack files",
    )
    parser.add_argument(
        "--object", type=str, required=True, help="Object name: e.g. 035_power_drill"
    )
    args = parser.parse_args()

    align_data(data_path=args.data_path, object_name=args.object)
