
# Copyright (c) 2022 Carnegie Mellon University <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)

"""
Record DIGIT images and webcam
python record_data_expt.py --duration 60 --webcam_port 2 --object 035_power_drill
Note: connect the DIGIT after you connect the webcam, and modify webcam_port according to v4l2-ctl --list-devices
"""

from datetime import datetime
import os
import cv2
import sys
from digit_interface import Digit, DigitHandler
import time
import numpy as np
from threading import Thread
import argparse
import tqdm
from ycb_slide.utils.misc import current_time, datetime_to_str, datetime_diff


def grab_frame(d: Digit):
    """
    Grab image from DIGIT
    """
    return d.get_frame()


def make_data_folder(datafolder, object_name: str):
    """
    Create data folder to store experiment
    """

    now = datetime.now()
    date_dir = os.path.join(datafolder, now.strftime("%d_%m_%Y"))
    if not os.path.exists(os.path.join(date_dir, object_name)):
        os.makedirs(os.path.join(date_dir, object_name))

    # Input how many data frames to collect
    i = 0
    while os.path.exists(
        os.path.join(date_dir, object_name, "dataset_{}".format(str(i)))
    ):
        i += 1
    object_dir = os.path.join(date_dir, object_name, "dataset_{}".format(str(i)))
    os.makedirs(os.path.join(object_dir, "frames"))
    os.makedirs(os.path.join(object_dir, "webcam_frames"))

    # print('Saving data to: {}'.format(object_dir))
    return object_dir


class webcam_thread(object):
    """
    Capture webcam data in a separate thread
    """

    def __init__(self, cam_id=0):
        self.capture = cv2.VideoCapture(cam_id)
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 10)

        if not self.capture.isOpened():
            print("Port error, capture not opened!")
            sys.exit(1)

        self.frame = None
        # https://www.kurokesu.com/main/2020/07/12/pulling-full-resolution-from-a-webcam-with-opencv-windows/
        self.capture.set(cv2.CAP_PROP_FPS, 30.0)
        self.capture.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("m", "j", "p", "g")
        )
        self.capture.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G")
        )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            # time.sleep(self.FPS)

    def grab_frame(self):
        return self.frame
        # cv2.imshow('frame', self.frame)
        # cv2.waitKey(self.FPS_MS)


def connect_digit(resolution="QVGA"):
    """
    Initialize the DIGIT capture
    """
    try:
        connected_digit = DigitHandler.list_digits()[0]
    except:
        print("No DIGIT found!")
        sys.exit(1)
    d = Digit(connected_digit["serial"])  # Unique serial number
    d.connect()
    d.set_resolution(Digit.STREAMS[resolution])
    print(d.info())
    # print("Collecting data from DIGIT {}".format(d.serial))
    return d


def record_data(obj: str, max_time: float = 120.0, cam_id: int = 4):
    """
    Capture DIGIT frames for max_time with webcam
    """
    try:
        d = connect_digit()
        cap = webcam_thread(cam_id=cam_id)
    except:
        print("cam_id: {} not recognized".format(cam_id))
        return

    object_dir = make_data_folder(datafolder="data", object_name=obj)

    for _ in tqdm.tqdm(range(100)):
        # grab a few frames for stability (10 secs)
        time.sleep(0.1)
        grab_frame(d), cap.grab_frame()

    start_time, curr_time = current_time(), 0.0
    digit_data = {
        "start_time": datetime_to_str(start_time),
        "digit_frames": [],
        "webcam_frames": [],
        "timestamps": [],
    }
    digit_frames, webcam_frames = [], []

    # Run data collection until time expires or keyboard interrupt
    while curr_time < max_time:
        digit_frame, webcam_frame = grab_frame(d), cap.grab_frame()

        # TODO: diff times
        curr_time = datetime_diff(current_time(), start_time)

        height, width, _ = digit_frame.shape
        cv2.imshow(
            "DIGIT {}".format(d.serial),
            cv2.resize(digit_frame, (width * 2, height * 2)),
        )
        cv2.waitKey(1)

        if len(digit_frames) % 100 == 0:
            print("Time: {:.1f}".format(curr_time))
        # print('Saving image {}'.format(count))
        digit_frames.append(digit_frame)
        webcam_frames.append(webcam_frame)
        digit_data["timestamps"].append(curr_time)

    cv2.destroyAllWindows()
    d.disconnect()

    print("{} seconds complete: saving data to {}".format(max_time, object_dir))
    for i, (digit_frame, webcam_frame) in enumerate(zip(digit_frames, webcam_frames)):
        filename = "frames/frame_{}.jpg".format(str(i).zfill(7))
        webcam_filename = "webcam_frames/frame_{}.jpg".format(str(i).zfill(7))
        digit_data["digit_frames"].append(filename)
        digit_data["webcam_frames"].append(webcam_filename)
        cv2.imwrite(os.path.join(object_dir, filename), digit_frame)
        cv2.imwrite(os.path.join(object_dir, webcam_filename), webcam_frame)

    np.save(os.path.join(object_dir, "digit_data.npy"), digit_data)
    print("Saved numpy data: {}".format(os.path.join(object_dir, "digit_data.npy")))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record DIGIT data and webcam frames")
    parser.add_argument(
        "--duration", type=float, required=True, help="Duration of recording"
    )
    parser.add_argument(
        "--webcam_port", type=int, required=True, help="Port to webcam (e.g. 2, 4, 5"
    )
    parser.add_argument(
        "--object", type=str, required=True, help="Object name: e.g. 035_power_drill"
    )
    args = parser.parse_args()

    # change to cwd
    abspath = os.path.abspath(__file__ + "/../")
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    record_data(obj=args.object, max_time=args.duration, cam_id=args.webcam_port)
