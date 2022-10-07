# Copyright (c) 2022 Carnegie Mellon University <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)

"""
Visualizes the sim/real dataset poses 

Real data example: 
python visualize_trajectory.py --data_path dataset/real/035_power_drill/dataset_0 --object 035_power_drill --real
Sim data example:
python visualize_trajectory.py --data_path dataset/sim/035_power_drill/00 --object 035_power_drill
"""

import os
from os import path as osp
import numpy as np
from ycb_slide.utils.misc import (
    viz_poses_pointclouds_on_mesh,
    process_real_poses,
    process_sim_poses,
)
import argparse


def visualize_trajectory(data_path, object_name, real):
    dname = osp.dirname(osp.abspath(__file__))
    os.chdir(dname)
    os.chdir("../..")  # root
    print(f"Object: {object_name}, Dataset: {data_path}\n")

    obj_path = osp.join("dataset", "obj_models", object_name, "nontextured.stl")

    if not osp.isdir(data_path):
        print(f"No dataset {data_path} found")
        return

    if real:
        # image_path = osp.join(data_path, "frames")
        pose_file = osp.join(data_path, "synced_data.npy")
        alignment_file = osp.join(data_path, "..", "alignment.npy")
        digit_poses = process_real_poses(pose_file, alignment_file, object_name)
        print("Visualizing real data")
    else:
        # image_path = osp.join(data_path, "tactile_images")
        pose_file = osp.join(data_path, "tactile_data.pkl")
        digit_poses = process_sim_poses(pose_file)
        print(f"Visualizing sim data")

    viz_poses_pointclouds_on_mesh(
        mesh_path=obj_path,
        poses=digit_poses,
        pointclouds=[],
        save_path=None,
        decimation_factor=None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trajectory")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset",
    )
    parser.add_argument(
        "--object", type=str, required=True, help="Object name: e.g. 035_power_drill"
    )
    parser.add_argument("--real", action="store_true", default=False)

    args = parser.parse_args()

    visualize_trajectory(
        data_path=args.data_path, object_name=args.object, real=args.real
    )
