# Copyright (c) 2022 Carnegie Mellon University <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)

"""
Timesync optitrack .csv and DIGIT data .npy files
python timesync_digit.py --digit_file dataset/real/035_power_drill/dataset_0/digit_data.npy --tracking_file dataset/real/035_power_drill/035_power_drill.csv --bodies "DIGIT 035_power_drill"
"""

import os
import argparse
from ycb_slide.utils.optitrack import parse_optitrack_csv, plot_optitrack_poses
from ycb_slide.utils.misc import sync_optitrack_digit
import numpy as np


def timesync_data(digit_file: str, tracking_file: str, bodies: list):
    digit_data = np.load(
        digit_file, allow_pickle=True
    ).item()  # dict_keys(['frame', 'timestamp', 'image_path', 'webcam_path', 'start_time'])
    tracking_data = parse_optitrack_csv(csvFile=tracking_file, bodies=bodies)
    synced_data = sync_optitrack_digit(tracking_data, digit_data)
    plot_optitrack_poses(
        body_poses=synced_data["poses"],
        save_path=digit_file.replace("digit_data.npy", ""),
    )
    np.save(digit_file.replace("digit_data", "synced_data"), synced_data)
    print(
        "Saved synced numpy data: {}".format(
            digit_file.replace("digit_data", "synced_data")
        )
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Timesync optitrack .csv and DIGIT data .npy files"
    )
    parser.add_argument(
        "--digit_file", type=str, required=True, help="Path to DIGIT data"
    )
    parser.add_argument(
        "--tracking_file", type=str, required=True, help="Path to optitrack data file"
    )
    parser.add_argument(
        "--bodies", type=str, nargs="*", required=True, help="bodies to track"
    )
    args = parser.parse_args()

    # change to one directory up
    abspath = os.path.abspath(__file__ + "/../")
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    timesync_data(
        digit_file=args.digit_file, tracking_file=args.tracking_file, bodies=args.bodies
    )
