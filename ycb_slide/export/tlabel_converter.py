# Copyright (c) 2022 Carnegie Mellon University <suddhu@cmu.edu>
# This code is licensed under MIT license (see LICENSE.txt for details)
"""
Convert YCB-Slide dataset to TLabel format.

TLabel is an open Tactile Label standard for cross-dataset interoperability
(see https://github.com/liu-luo/tlabel). This converter produces .tlabel.json
files that can be loaded by any TLabel-compatible tool for visualization,
annotation, and downstream processing.

Usage:
    # Convert all real-world objects
    python -m ycb_slide.export.tlabel_converter --data_path dataset/real --split real

    # Convert a specific object
    python -m ycb_slide.export.tlabel_converter --data_path dataset/real --split real --object 035_power_drill

    # Convert simulated data
    python -m ycb_slide.export.tlabel_converter --data_path dataset/sim --split sim --object 035_power_drill

    # Convert all objects in both splits
    python -m ycb_slide.export.tlabel_converter --data_path dataset --split all
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime


def convert_real_sequence(dataset_dir, object_name, dataset_id):
    """Convert one real-world dataset (dataset_X) to a TLabel sequence.

    Reads synced_data.npy which contains:
        - timestamps: synchronized timestamps (list[float])
        - digit_frames: relative paths to DIGIT images (list[str])
        - webcam_frames: relative paths to webcam images (list[str])
        - poses: dict mapping rigid body names to (N,7) arrays [x,y,z,qx,qy,qz,qw]

    Args:
        dataset_dir: Path to a dataset_X directory
        object_name: Name of the YCB object (e.g. '035_power_drill')
        dataset_id: Integer dataset index

    Returns:
        dict: TLabel sequence dictionary
    """
    synced_path = dataset_dir / "synced_data.npy"
    if not synced_path.exists():
        raise FileNotFoundError(f"synced_data.npy not found in {dataset_dir}")

    synced = np.load(str(synced_path), allow_pickle=True).item()

    timestamps = synced.get("timestamps", [])
    digit_frames = synced.get("digit_frames", [])
    webcam_frames = synced.get("webcam_frames", [])
    poses = synced.get("poses", {})

    digit_pose = poses.get("DIGIT", np.array([]))
    obj_keys = [k for k in poses.keys() if k != "DIGIT"]
    obj_pose = poses.get(obj_keys[0], np.array([])) if obj_keys else np.array([])

    num_frames = len(timestamps)

    # Build per-frame data
    frames = []
    for i in range(num_frames):
        frame = {
            "frame_idx": i,
            "timestamp_s": float(timestamps[i]),
            "images": {},
            "poses": {},
        }

        # DIGIT image path
        if i < len(digit_frames):
            frame_path = dataset_dir / digit_frames[i]
            if frame_path.exists():
                frame["images"]["digit"] = str(frame_path)

        # Webcam image path
        if i < len(webcam_frames):
            wc_path = dataset_dir / webcam_frames[i]
            if wc_path.exists():
                frame["images"]["webcam"] = str(wc_path)

        # DIGIT sensor pose
        if len(digit_pose) > i:
            p = digit_pose[i]
            frame["poses"]["sensor_pose"] = {
                "position": p[:3].tolist(),
                "orientation": p[3:7].tolist(),  # qx, qy, qz, qw
            }

        # Object pose
        if len(obj_pose) > i:
            p = obj_pose[i]
            frame["poses"]["object_pose"] = {
                "position": p[:3].tolist(),
                "orientation": p[3:7].tolist(),  # qx, qy, qz, qw
            }

        frames.append(frame)

    seq = {
        "seq_id": f"{object_name}_dataset{dataset_id}",
        "sensor_profile": {
            "name": "DIGIT",
            "sensor_type": "vision_based",
            "resolution": [240, 320],
            "frame_rate_hz": 30.0,
            "raw_unit": "pixel",
            "physical_semantics": {
                "modality": "vision_based_tactile",
                "measurement_type": "elastomer_deformation",
            },
        },
        "metadata": {
            "source_dataset": "YCB-Slide",
            "split": "real",
            "object_name": object_name,
            "dataset_id": dataset_id,
            "num_frames": num_frames,
            "duration_s": float(timestamps[-1] - timestamps[0]) if num_frames > 0 else 0.0,
        },
        "frames": frames,
    }

    return seq


def convert_sim_sequence(traj_dir, object_name, traj_id):
    """Convert one simulated trajectory (e.g. 00/) to a TLabel sequence.

    Reads tactile_data.pkl which contains:
        - gelposes_meas: (N, 7) measured sensor poses [x,y,z,qx,qy,qz,qw]
        - gelposes: (N, 7) ground-truth sensor poses
        - camposes: (N, 7) camera poses

    Args:
        traj_dir: Path to a trajectory directory (e.g. sim/004_sugar_box/00)
        object_name: Name of the YCB object
        traj_id: Integer trajectory index

    Returns:
        dict: TLabel sequence dictionary
    """
    import dill as pickle

    pkl_path = traj_dir / "tactile_data.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"tactile_data.pkl not found in {traj_dir}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    gelposes_meas = np.array(data.get("gelposes_meas", []))
    gelposes = np.array(data.get("gelposes", []))
    num_frames = len(gelposes_meas)

    # Tactile image paths (numbered sequentially)
    img_dir = traj_dir / "tactile_images"

    frames = []
    for i in range(num_frames):
        frame = {
            "frame_idx": i,
            "timestamp_s": i / 30.0,  # sim runs at 30Hz
            "images": {},
            "poses": {},
        }

        # Tactile image
        img_path = img_dir / f"{i}.jpg"
        if img_path.exists():
            frame["images"]["digit"] = str(img_path)

        # Measured sensor pose (with noise)
        if i < len(gelposes_meas):
            p = gelposes_meas[i]
            frame["poses"]["sensor_pose_noisy"] = {
                "position": p[:3].tolist(),
                "orientation": p[3:7].tolist(),
            }

        # Ground-truth sensor pose
        if i < len(gelposes):
            p = gelposes[i]
            frame["poses"]["sensor_pose_gt"] = {
                "position": p[:3].tolist(),
                "orientation": p[3:7].tolist(),
            }

        frames.append(frame)

    seq = {
        "seq_id": f"{object_name}_{traj_id:02d}",
        "sensor_profile": {
            "name": "DIGIT",
            "sensor_type": "vision_based",
            "resolution": [240, 320],
            "frame_rate_hz": 30.0,
            "raw_unit": "pixel",
            "physical_semantics": {
                "modality": "vision_based_tactile",
                "measurement_type": "elastomer_deformation",
            },
        },
        "metadata": {
            "source_dataset": "YCB-Slide",
            "split": "sim",
            "object_name": object_name,
            "trajectory_id": traj_id,
            "num_frames": num_frames,
            "duration_s": num_frames / 30.0,
            "sim_noise": {
                "sigma_trans_mm": 0.5,
                "sigma_rot_deg": 1.0,
            },
        },
        "frames": frames,
    }

    return seq


def convert_all(data_path, split="real", object_filter=None, output=None):
    """Convert YCB-Slide data to TLabel format.

    Args:
        data_path: Root path to YCB-Slide dataset
        split: 'real', 'sim', or 'all'
        object_filter: Optional object name to convert only one object
        output: Output file path (default: auto-generated)
    """
    data_root = Path(data_path)
    all_sequences = []

    splits_to_process = []
    if split in ("real", "all"):
        real_dir = data_root / "real" if split == "all" else data_root
        if real_dir.exists():
            splits_to_process.append(("real", real_dir))
    if split in ("sim", "all"):
        sim_dir = data_root / "sim" if split == "all" else data_root
        if sim_dir.exists():
            splits_to_process.append(("sim", sim_dir))

    for split_name, split_dir in splits_to_process:
        # Find object directories
        if object_filter:
            obj_dirs = [split_dir / object_filter]
            obj_dirs = [d for d in obj_dirs if d.exists()]
        else:
            obj_dirs = sorted([
                d for d in split_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

        print(f"\n{'='*60}")
        print(f"Converting [{split_name}] split: {len(obj_dirs)} object(s)")
        print(f"{'='*60}")

        for obj_dir in obj_dirs:
            object_name = obj_dir.name
            print(f"\n  --- {object_name} ---")

            if split_name == "real":
                # Real data: dataset_0, dataset_1, ...
                dataset_dirs = sorted([
                    d for d in obj_dir.iterdir()
                    if d.is_dir() and d.name.startswith("dataset_")
                ])
                for ds_dir in dataset_dirs:
                    ds_id = int(ds_dir.name.split("_")[1])
                    try:
                        seq = convert_real_sequence(ds_dir, object_name, ds_id)
                        all_sequences.append(seq)
                        print(f"    [OK] {ds_dir.name}: {seq['metadata']['num_frames']} frames")
                    except Exception as e:
                        print(f"    [FAIL] {ds_dir.name}: {e}")

            elif split_name == "sim":
                # Sim data: 00, 01, ...
                traj_dirs = sorted([
                    d for d in obj_dir.iterdir()
                    if d.is_dir() and d.name.isdigit()
                ])
                for traj_dir in traj_dirs:
                    traj_id = int(traj_dir.name)
                    try:
                        seq = convert_sim_sequence(traj_dir, object_name, traj_id)
                        all_sequences.append(seq)
                        print(f"    [OK] {traj_dir.name}: {seq['metadata']['num_frames']} frames")
                    except Exception as e:
                        print(f"    [FAIL] {traj_dir.name}: {e}")

    # Build output
    result = {
        "metadata": {
            "format_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "source_dataset": "YCB-Slide",
            "source_url": "https://github.com/rpl-cmu/YCB-Slide",
            "paper": "MidasTouch: Monte-Carlo inference over distributions across sliding touch (CoRL 2022)",
            "num_sequences": len(all_sequences),
            "sensor": "DIGIT",
            "task": "Sliding tactile interaction with pose ground truth",
        },
        "sequences": all_sequences,
    }

    # Determine output path
    if output is None:
        output = str(data_root / f"ycb_slide_{split}.tlabel.json")

    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    total_frames = sum(len(s["frames"]) for s in all_sequences)
    file_size_mb = os.path.getsize(output) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"Output : {output}")
    print(f"Size   : {file_size_mb:.1f} MB")
    print(f"Objects: {len(obj_dirs)}")
    print(f"Seqs   : {len(all_sequences)}")
    print(f"Frames : {total_frames}")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YCB-Slide dataset to TLabel format"
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Root path to YCB-Slide dataset (containing real/ and/or sim/)"
    )
    parser.add_argument(
        "--split", type=str, default="real", choices=["real", "sim", "all"],
        help="Which split to convert (default: real)"
    )
    parser.add_argument(
        "--object", type=str, default=None,
        help="Convert only a specific object (e.g. 035_power_drill)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output .tlabel.json path (default: <data_path>/ycb_slide_<split>.tlabel.json)"
    )
    args = parser.parse_args()

    convert_all(
        data_path=args.data_path,
        split=args.split,
        object_filter=args.object,
        output=args.output,
    )
