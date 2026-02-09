"""
Isaac Sim dataset class for OpenGraph.

This is a modified version of SemanticKittiDataset that does NOT apply
the T_cam2_velo transform to poses. This matches how opennav_mem handles
Isaac Sim data in datasets_class_IsaacROS.py.

The key difference from SemanticKITTI:
- SemanticKITTI stores CAMERA poses, which need to be transformed to velodyne frame
- Isaac Sim (via collect_isaac_data.py) stores BASE_LINK poses directly

The Tr matrix is still loaded and used for point projection (P @ Tr @ points),
but NOT applied to poses.
"""

import glob
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from natsort import natsorted


class IsaacDataset(torch.utils.data.Dataset):
    """Dataset class for Isaac Sim data collected by collect_isaac_data.py."""

    def __init__(
        self,
        basedir: Union[Path, str],
        sequence: Union[Path, str],
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.calib_path = os.path.join(self.input_folder, "calib.txt")
        self.calib = self.load_calib()
        self.pose_path = os.path.join(self.input_folder, "poses.txt")
        self.poses = self.load_poses()
        self.color_paths = natsorted(glob.glob(f"{self.input_folder}/image_2/*.png"))
        self.pc_paths = natsorted(glob.glob(f"{self.input_folder}/velodyne/*.bin"))

        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end == -1 or end > start):
            raise ValueError(
                "end ({0}) must be -1 (use all images) or greater than start ({1})".format(end, start)
            )
        if len(self.color_paths) != len(self.pc_paths):
            raise ValueError("Number of color and point cloud files must be the same.")

        self.num_imgs = len(self.color_paths)
        if self.end == -1:
            self.end = self.num_imgs

        # Keep all poses and pc_paths for historical frame overlap
        self.all_pc_paths = self.pc_paths
        self.all_poses = self.poses

        # Apply stride
        self.stride = stride
        self.color_paths = self.color_paths[self.start : self.end : stride]
        self.pc_paths = self.pc_paths[self.start : self.end : stride]
        self.poses = self.poses[self.start : self.end : stride]
        self.num_imgs = len(self.color_paths)

        print(f"\n Isaac Sim dataset loaded: {self.num_imgs} frames from {self.input_folder}\n")
        super().__init__()

    def load_poses(self):
        """Load poses WITHOUT applying T_cam2_velo transform.

        This is the key difference from SemanticKittiDataset.
        Isaac data stores base_link poses directly (not camera poses),
        so no additional transform is needed.

        This matches opennav_mem's datasets_class_IsaacROS.py line 198:
            poses = poses# @ self.calib['T_cam2_velo']  (commented out)
        """
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
            poses = np.array([list(map(float, line.strip().split())) for line in lines])
            poses = poses.reshape(-1, 3, 4)
            ones_column = np.zeros((poses.shape[0], 1, 4))
            ones_column[:, :, -1] = 1.0
            poses = np.append(poses, ones_column, axis=1)
            # NOTE: Unlike SemanticKittiDataset, we do NOT apply T_cam2_velo here
            # because Isaac poses are already in base_link frame
        return poses

    def load_calib(self):
        """Load calibration file (same as SemanticKittiDataset)."""
        calib = {}
        with open(self.calib_path, "r") as calib_file:
            calib_lines = calib_file.readlines()
            # Load camera intrinsics (P2 line)
            P_rect_line = calib_lines[2]
            P_rect_02 = np.array(list(map(float, P_rect_line.strip().split()[1:]))).reshape(3, 4)
            calib["P_rect_20"] = P_rect_02
            # Load camera extrinsics (Tr line)
            Tr_line = calib_lines[4]
            Tr = np.array(list(map(float, Tr_line.strip().split()[1:]))).reshape(3, 4)
            Tr = np.vstack([Tr, [0, 0, 0, 1]])
            calib['T_cam2_velo'] = Tr
        return calib

    def load_velo_scan(self, velo_filename):
        """Load point cloud from binary file."""
        scan = np.fromfile(velo_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return scan

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        """Get a single frame: image, point cloud, pose, and historical data."""
        color_path = self.color_paths[index]
        pc_path = self.pc_paths[index]
        color = cv2.imread(color_path)
        pointCloud = self.load_velo_scan(pc_path)
        pose = self.poses[index]

        # Historical frames for multi-frame overlap projection
        his_pointCloud = []
        his_pose = []
        if index > 0:
            for i in range(self.stride - 1):
                his_index = self.start + index * self.stride - i - 1
                if his_index >= 0 and his_index < len(self.all_pc_paths):
                    his_pointCloud.append(self.load_velo_scan(self.all_pc_paths[his_index]))
                    his_pose.append(self.all_poses[his_index])

        return (
            color,
            pointCloud,
            pose,
            his_pointCloud,
            his_pose
        )
