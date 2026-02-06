#!/usr/bin/env python3
"""
Verify collected Isaac Sim data is correctly formatted for OpenGraph.

Loads data using SemanticKittiDataset, projects point clouds onto images,
and saves overlay visualizations to confirm calibration is correct.

Usage (run in opengraph-cu118 conda env):
    python script/verify_collection.py \
        --basedir data/isaac_warehouse \
        --sequence 01 \
        --num_frames 5
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from some_class.datasets_class import SemanticKittiDataset


def project(points, image, calib):
    """Project point cloud onto image (adapted from OpenGraph utils.py).

    Points are (N, 4) with [x, y, z, intensity]. We use only xyz for projection.
    """
    # Extract xyz and create homogeneous coordinates (N, 4) -> (4, N)
    xyz = points[:, :3]  # Take only x, y, z
    ones = np.ones((xyz.shape[0], 1))
    points_homo = np.hstack([xyz, ones]).T  # (4, N)

    # Filter points behind the sensor (x < 0 in velodyne frame)
    valid_mask = points_homo[0, :] >= 0
    points_homo = points_homo[:, valid_mask]
    pointCloud = points[valid_mask]

    # Project: P_rect @ T_cam2_velo @ points_homo
    proj_lidar = calib["P_rect_20"].dot(calib["T_cam2_velo"]).dot(points_homo)

    # Filter points behind camera (z < 0 in camera frame)
    valid_mask = proj_lidar[2, :] > 0
    cam = proj_lidar[:, valid_mask]
    pointCloud = pointCloud[valid_mask]

    # Perspective division
    cam[:2, :] /= cam[2, :]

    # Filter points outside image bounds
    IMG_H, IMG_W, _ = image.shape
    u, v, z = cam
    valid_mask = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    cam = cam[:, valid_mask]
    pointCloud = pointCloud[valid_mask]

    u, v, z = cam
    pixels = np.stack([v, u], axis=-1)  # (N, 2) with [row, col]
    return pointCloud, pixels


def main():
    parser = argparse.ArgumentParser(description="Verify collected data format")
    parser.add_argument("--basedir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "isaac_warehouse"
                        ))
    parser.add_argument("--sequence", type=str, default="01")
    parser.add_argument("--num_frames", type=int, default=5,
                        help="Number of frames to verify")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to save overlay images (default: {basedir}/{seq}/verify/)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.basedir, args.sequence, "verify")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from: {args.basedir}/{args.sequence}")
    try:
        dataset = SemanticKittiDataset(
            basedir=args.basedir,
            sequence=args.sequence,
            stride=1,
            start=0,
            end=-1,
        )
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        print("Check that the directory contains: image_2/, velodyne/, poses.txt, calib.txt")
        sys.exit(1)

    print(f"Dataset loaded: {len(dataset)} frames")
    print(f"Calibration P_rect_20:\n{dataset.calib['P_rect_20']}")
    print(f"Calibration T_cam2_velo:\n{dataset.calib['T_cam2_velo']}")

    # Verify sample frames
    step = max(1, len(dataset) // args.num_frames)
    indices = list(range(0, len(dataset), step))[:args.num_frames]

    for idx in indices:
        color, pc, pose, his_pcs, his_poses = dataset[idx]
        print(f"\n--- Frame {idx} ---")
        print(f"  Image shape: {color.shape}")
        print(f"  Point cloud shape: {pc.shape}")
        print(f"  Point cloud range: x=[{pc[:,0].min():.2f}, {pc[:,0].max():.2f}], "
              f"y=[{pc[:,1].min():.2f}, {pc[:,1].max():.2f}], "
              f"z=[{pc[:,2].min():.2f}, {pc[:,2].max():.2f}]")
        print(f"  Pose translation: [{pose[0,3]:.2f}, {pose[1,3]:.2f}, {pose[2,3]:.2f}]")
        print(f"  Historical frames: {len(his_pcs)}")

        # Project point cloud onto image
        try:
            proj_points, pixels = project(pc, color, dataset.calib)
        except Exception as e:
            print(f"  WARNING: Projection failed: {e}")
            continue

        print(f"  Projected points: {len(proj_points)} (of {pc.shape[0]})")

        if len(proj_points) == 0:
            print("  WARNING: No points projected onto image! Check calibration.")
            continue

        # Create overlay visualization
        overlay = color.copy()
        depths = proj_points[:, 2] if proj_points.shape[1] > 2 else np.ones(len(proj_points))
        depth_norm = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)

        for i in range(min(len(pixels), 50000)):
            v, u = int(pixels[i, 0]), int(pixels[i, 1])
            if 0 <= v < overlay.shape[0] and 0 <= u < overlay.shape[1]:
                # Color by depth: blue (near) → red (far)
                b = int(255 * (1 - depth_norm[i]))
                r = int(255 * depth_norm[i])
                cv2.circle(overlay, (u, v), 1, (b, 0, r), -1)

        # Save overlay
        out_path = os.path.join(args.output_dir, f"verify_{idx:06d}.png")
        cv2.imwrite(out_path, overlay)
        print(f"  Saved overlay: {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"  Total frames:  {len(dataset)}")
    print(f"  Overlays saved to: {args.output_dir}")
    print(f"\n  If the projected points align with objects in the images,")
    print(f"  the calibration and data format are correct.")
    print(f"\n  Next steps:")
    print(f"    1. python script/main_gen_cap.py --config-name=isaac_warehouse")
    print(f"    2. torchrun --nproc_per_node=1 script/main_gen_pc.py --config-name=isaac_warehouse")
    print(f"    3. python script/build_scenegraph.py --config-name=isaac_warehouse")
    print("=" * 60)


if __name__ == "__main__":
    main()
