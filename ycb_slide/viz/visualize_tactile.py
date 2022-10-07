# Copyright (c) 2022 Sudharshan Suresh <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)

"""
Visualizes the tactile images (real/sim) + ground-truth data (sim only)

Real data example: 
python visualize_tactile.py --data_path dataset/real/035_power_drill/dataset_0 --object 035_power_drill --real
Sim data example:
python visualize_tactile.py --data_path dataset/sim/035_power_drill/00 --object 035_power_drill
"""

import os
from os import path as osp
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def visualize_tactile(data_path, object_name, real):
    dname = osp.dirname(osp.abspath(__file__))
    os.chdir(dname)
    os.chdir("../..")  # root
    print(f"Object: {object_name}, Dataset: {data_path}\n")

    if not osp.isdir(data_path):
        print(f"No dataset {data_path} found")
        return

    if real:
        image_path = osp.join(data_path, "frames")
        data_type = "real"
        viz_mosaic = [["Tactile image"]]
        images = sorted(os.listdir(image_path))

    else:
        image_path = osp.join(data_path, "tactile_images")
        heightmap_path, contactmask_path = osp.join(
            data_path, "gt_heightmaps"
        ), osp.join(data_path, "gt_contactmasks")
        data_type = "sim"
        viz_mosaic = [
            ["Tactile image", "GT heightmap", "GT contact mask"],
        ]

        # load images and ground truth depthmaps
        images = sorted(os.listdir(image_path), key=lambda y: int(y.split(".")[0]))
        heightmaps = sorted(os.listdir(heightmap_path), key=lambda y: int(y.split(".")[0]))
        contact_masks = sorted(
            os.listdir(contactmask_path), key=lambda y: int(y.split(".")[0])
        )
        images = sorted(os.listdir(image_path), key=lambda y: int(y.split(".")[0]))

    plt.ion()
    fig, axes = plt.subplot_mosaic(mosaic=viz_mosaic, constrained_layout=True)
    fig.suptitle(f"{object_name} {data_type} data", fontsize=16)

    N = len(images)
    for i in range(N):
        # Open images
        image = np.array(Image.open(osp.join(image_path, images[i])))
        viz_data = [image / 255.0]

        if not real:
            gt_heightmap = np.array(
                Image.open(osp.join(heightmap_path, heightmaps[i]))
            ).astype(np.int64)
            contactmask = np.array(
                Image.open(osp.join(contactmask_path, contact_masks[i]))
            ).astype(bool)

            viz_data += [gt_heightmap, contactmask]

        for j, (label, ax) in enumerate(axes.items()):
            ax.clear()
            ax.imshow(viz_data[j])
            ax.axis("off")
            ax.set_title(f'{label} {i}' )
        plt.pause(0.05)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize tactile data")
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

    visualize_tactile(data_path=args.data_path, object_name=args.object, real=args.real)
