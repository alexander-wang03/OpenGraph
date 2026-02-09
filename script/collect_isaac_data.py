#!/usr/bin/env python3
"""
Isaac Sim → SemanticKITTI Data Collector

Subscribes to ROS2 topics published by Isaac Sim (RGB camera, depth/LiDAR, TF)
and saves synchronized frames to SemanticKITTI format for offline processing
with OpenGraph.

Output structure:
    {output_dir}/{sequence}/
        image_2/000000.png, 000001.png, ...
        velodyne/000000.bin, 000001.bin, ...
        poses.txt
        calib.txt
        time.txt

Usage:
    # Depth mode (RealSense D455 depth camera):
    python3 collect_isaac_data.py --mode depth \
        --rgb_topic /camera/color/image_raw \
        --depth_topic /camera/depth/image_raw

    # LiDAR mode:
    python3 collect_isaac_data.py --mode lidar \
        --rgb_topic /camera/color/image_raw \
        --lidar_topic /ouster_lidar/point_cloud

    # With navigation (use sim time from Isaac Sim):
    python3 collect_isaac_data.py --mode depth --use_sim_time
"""

import argparse
import os
import sys
import time
import signal

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from cv_bridge import CvBridge
import tf_transformations
import message_filters

try:
    import sensor_msgs_py.point_cloud2 as pc2
except ImportError:
    from sensor_msgs import point_cloud2 as pc2


# ---------------------------------------------------------------------------
# Validated calibration for Isaac Sim Husky RealSense D455 @ 1280x720
# Source: /home/awang/Documents/TRAILbot/opennav_mem/star/config/calib.txt
# ---------------------------------------------------------------------------
DEFAULT_P2 = np.array([
    [634.086240, 0.0,        640.0, 0.0],
    [0.0,        566.489981, 360.0, 0.0],
    [0.0,        0.0,        1.0,   0.0],
], dtype=np.float64)

# Camera intrinsics (3x3) extracted from P2
DEFAULT_K = DEFAULT_P2[:, :3]

# Velodyne→Camera transform (Tr line in SemanticKITTI calib.txt)
# For the Husky, this is approximately base_link → color_camera
# This is used for point projection: P2 @ Tr @ points_velodyne = pixels
DEFAULT_TR = np.array([
    [-5.960463944632e-08, -1.0,                7.232996068751e-08, -1.500106569601e-03],
    [-3.162113404453e-08, -7.232995891115e-08, -1.0,               -3.409403746934e-01],
    [ 1.0,                -5.960464138921e-08, -3.162112993671e-08, -4.399998542302e-01],
], dtype=np.float64)

# base_link → camera as 4x4 (for transforming points from camera to base_link frame)
T_BASELINK_TO_CAM = np.vstack([DEFAULT_TR, [0, 0, 0, 1]])

# For poses: we store base_link poses directly, so Tr for pose transformation should be identity.
# But we still need the real Tr for point projection (P2 @ Tr @ points).
# OpenGraph's load_poses() does: poses = poses @ T_cam2_velo
# If we store base_link poses and want them unchanged, we need Tr = identity for that multiplication.
# However, Tr is also used for projection. So we have a conflict.
#
# Solution: Store base_link poses. The Tr will transform them, but since our points are
# already in base_link frame, the projection P2 @ Tr @ points_baselink will work correctly
# because Tr transforms base_link→camera, matching what P2 expects.


def depth_to_pointcloud(depth_img, K, subsample=4, max_depth=15.0):
    """Backproject depth image to 3D point cloud in camera frame.

    Args:
        depth_img: (H, W) float32 depth in meters.
        K: (3, 3) camera intrinsics.
        subsample: Take every N-th pixel to reduce density.
        max_depth: Ignore points beyond this distance.

    Returns:
        points: (N, 4) float32 array [x, y, z, intensity=1.0].
    """
    H, W = depth_img.shape
    u, v = np.meshgrid(
        np.arange(0, W, subsample),
        np.arange(0, H, subsample),
    )
    z = depth_img[::subsample, ::subsample].astype(np.float32)

    valid = (z > 0.01) & (z < max_depth) & np.isfinite(z)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x = (u[valid] - cx) * z[valid] / fx
    y = (v[valid] - cy) * z[valid] / fy
    z_valid = z[valid]

    points_xyz = np.stack([x, y, z_valid], axis=-1)
    intensity = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    points = np.hstack([points_xyz, intensity])  # (N, 4)
    return points


def format_calib(P2, Tr):
    """Format calibration matrices into SemanticKITTI calib.txt content."""
    zeros_12 = " ".join(["0.000000000000e+00"] * 12)
    p2_str = " ".join(f"{v:.6e}" for v in P2.flatten())
    tr_str = " ".join(f"{v:.12e}" for v in Tr.flatten())
    return (
        f"P0: {zeros_12}\n"
        f"P1: {zeros_12}\n"
        f"P2: {p2_str}\n"
        f"P3: {zeros_12}\n"
        f"Tr: {tr_str}\n"
    )


class IsaacDataCollector(Node):
    """ROS2 node that collects synchronized sensor data and saves to disk."""

    def __init__(self, args):
        super().__init__("isaac_data_collector")
        self.args = args
        self.bridge = CvBridge()
        self.frame_count = 0
        self.poses = []
        self.timestamps = []
        self.shutting_down = False
        self.last_save_time = 0.0
        self.save_interval = 1.0 / args.save_hz

        # TF state
        self.latest_tf = None

        # First pose for computing relative poses (like opennav_mem does)
        # This makes poses relative to starting position, avoiding drift issues
        self.first_pose = None
        self.first_pose_inv = None

        # Output directories
        self.seq_dir = os.path.join(args.output_dir, args.sequence)
        self.img_dir = os.path.join(self.seq_dir, "image_2")
        self.vel_dir = os.path.join(self.seq_dir, "velodyne")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.vel_dir, exist_ok=True)

        self.get_logger().info(f"Output directory: {self.seq_dir}")
        self.get_logger().info(f"Mode: {args.mode}")
        self.get_logger().info(f"Save frequency: {args.save_hz} Hz")

        # --- TF subscriber (always needed) ---
        self.tf_sub = self.create_subscription(
            TFMessage, "/tf", self.tf_callback, 10
        )
        # Internal publisher for synced base_link TF
        self.tf_pub = self.create_publisher(
            TransformStamped, "base_link_transform", 10
        )

        # --- Sensor subscribers ---
        self.image_sub = message_filters.Subscriber(
            self, Image, args.rgb_topic
        )
        self.pose_sub = message_filters.Subscriber(
            self, TransformStamped, "base_link_transform"
        )

        if args.mode == "lidar":
            self.sensor_sub = message_filters.Subscriber(
                self, PointCloud2, args.lidar_topic
            )
            ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.sensor_sub, self.pose_sub],
                queue_size=1000,
                slop=0.05,
            )
            ts.registerCallback(self.lidar_callback)
            self.get_logger().info(
                f"Subscribing: RGB={args.rgb_topic}, "
                f"LiDAR={args.lidar_topic}, TF=/tf"
            )

        elif args.mode == "depth":
            self.sensor_sub = message_filters.Subscriber(
                self, Image, args.depth_topic
            )
            ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.sensor_sub, self.pose_sub],
                queue_size=1000,
                slop=0.05,
            )
            ts.registerCallback(self.depth_callback)
            self.get_logger().info(
                f"Subscribing: RGB={args.rgb_topic}, "
                f"Depth={args.depth_topic}, TF=/tf"
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Keep reference so the synchronizer isn't garbage collected
        self._ts = ts

        self.get_logger().info("Waiting for sensor data...")

    # ------------------------------------------------------------------
    # TF handling
    # ------------------------------------------------------------------
    def tf_callback(self, msg):
        """Filter TF for base_link and republish for time-sync."""
        for transform in msg.transforms:
            if transform.child_frame_id == "base_link":
                tf_msg = TransformStamped()
                tf_msg.header.stamp = transform.header.stamp
                tf_msg.header.frame_id = "map"
                tf_msg.transform = transform.transform
                self.tf_pub.publish(tf_msg)
                self.latest_tf = transform

    def extract_pose(self, tf_msg):
        """Convert TransformStamped to 4x4 homogeneous matrix.

        Returns RELATIVE pose (relative to first frame) to avoid drift issues.
        This matches what opennav_mem does in run_data_collection.py:
            base_link_2map_TF = np.dot(first_transformation_matrix_inv, observation[3])
        """
        t = tf_msg.transform.translation
        r = tf_msg.transform.rotation
        quat = [r.x, r.y, r.z, r.w]
        rot_mat = tf_transformations.quaternion_matrix(quat)
        trans_mat = tf_transformations.translation_matrix([t.x, t.y, t.z])
        absolute_pose = np.dot(trans_mat, rot_mat)

        # Store first pose and compute its inverse
        if self.first_pose is None:
            self.first_pose = absolute_pose.copy()
            self.first_pose_inv = np.linalg.inv(self.first_pose)
            self.get_logger().info(
                f"First pose recorded at ({t.x:.2f}, {t.y:.2f}, {t.z:.2f})"
            )

        # Return relative pose: first frame will be identity
        relative_pose = np.dot(self.first_pose_inv, absolute_pose)
        return relative_pose

    # ------------------------------------------------------------------
    # Sensor callbacks
    # ------------------------------------------------------------------
    def lidar_callback(self, rgb_msg, lidar_msg, tf_msg):
        """Handle synchronized RGB + LiDAR + TF."""
        if self.shutting_down:
            return
        if not self._should_save():
            return

        # RGB image
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

        # LiDAR point cloud → (N, 4) with intensity
        point_gen = pc2.read_points(
            lidar_msg, field_names=("x", "y", "z"), skip_nans=True
        )
        points = np.array([point_gen["x"], point_gen["y"], point_gen["z"]]).T
        mask = ~np.all(points == 0, axis=1)
        points = points[mask]
        # Add intensity column (1.0) to match SemanticKITTI format
        intensity = np.ones((points.shape[0], 1), dtype=np.float32)
        points = np.hstack([points.astype(np.float32), intensity])

        # Store BASE_LINK pose (LiDAR points are already in base_link frame)
        T_world_base = self.extract_pose(tf_msg)

        self._save_frame(cv_image, points, T_world_base, rgb_msg.header.stamp)

    def depth_callback(self, rgb_msg, depth_msg, tf_msg):
        """Handle synchronized RGB + Depth + TF."""
        if self.shutting_down:
            return
        if not self._should_save():
            return

        # RGB image
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

        # Depth image
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth = depth.astype(np.float32)

        # Auto-detect depth units: if median > 100, likely millimeters
        valid_depth = depth[depth > 0]
        if len(valid_depth) > 0 and np.median(valid_depth) > 100.0:
            depth = depth / 1000.0  # Convert mm to meters
            if self.frame_count == 0:
                self.get_logger().info("Detected depth in millimeters, converting to meters")

        # Backproject to point cloud in camera frame
        points = depth_to_pointcloud(
            depth, DEFAULT_K,
            subsample=self.args.depth_subsample,
            max_depth=self.args.max_depth,
        )

        if points.shape[0] < 100:
            self.get_logger().warn(
                f"Very few points ({points.shape[0]}) from depth, skipping frame"
            )
            return

        # Transform points from camera frame to base_link frame.
        # This is our "velodyne" frame - where OpenGraph expects the points.
        # Projection will use P2 @ Tr @ points, where Tr = base_link→camera.
        T_cam_to_base = np.linalg.inv(T_BASELINK_TO_CAM)
        points_xyz = points[:, :3]
        points_homo = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])
        points_base = (T_cam_to_base @ points_homo.T).T[:, :3]
        points = np.hstack([points_base.astype(np.float32), points[:, 3:4]])

        # Store BASE_LINK pose in world frame (not camera pose).
        # OpenGraph's load_poses() multiplies by T_cam2_velo, which will give us
        # the pose in the "velodyne" frame. Since our velodyne=base_link, this is correct.
        # The point projection P2 @ Tr @ points works because points are in base_link
        # and Tr converts base_link→camera.
        T_world_base = self.extract_pose(tf_msg)

        self._save_frame(cv_image, points, T_world_base, rgb_msg.header.stamp)

    # ------------------------------------------------------------------
    # Save logic
    # ------------------------------------------------------------------
    def _should_save(self):
        """Throttle saves to target frequency."""
        now = time.time()
        if now - self.last_save_time < self.save_interval:
            return False
        self.last_save_time = now
        return True

    def _save_frame(self, cv_image, points, T_world_cam, stamp):
        """Save one frame to disk."""
        frame_id = f"{self.frame_count:06d}"

        # Save RGB image
        img_path = os.path.join(self.img_dir, f"{frame_id}.png")
        cv2.imwrite(img_path, cv_image)

        # Save point cloud as binary float32
        vel_path = os.path.join(self.vel_dir, f"{frame_id}.bin")
        points.astype(np.float32).tofile(vel_path)

        # Accumulate pose (upper 3x4 of camera-frame 4x4 matrix)
        self.poses.append(T_world_cam[:3, :].flatten())

        # Accumulate timestamp
        ts = stamp.sec + stamp.nanosec * 1e-9
        self.timestamps.append(ts)

        self.frame_count += 1
        if self.frame_count % 50 == 0:
            self.get_logger().info(
                f"Saved frame {self.frame_count} | "
                f"Points: {points.shape[0]} | "
                f"Image: {cv_image.shape[1]}x{cv_image.shape[0]}"
            )

        if self.args.max_frames > 0 and self.frame_count >= self.args.max_frames:
            self.get_logger().info(f"Reached max_frames={self.args.max_frames}, stopping")
            self.finalize()
            rclpy.shutdown()

    def finalize(self):
        """Write poses.txt, time.txt, and calib.txt, then print summary."""
        if self.shutting_down:
            return
        self.shutting_down = True

        # poses.txt: one line per frame, 12 floats (3x4 matrix flattened)
        poses_path = os.path.join(self.seq_dir, "poses.txt")
        with open(poses_path, "w") as f:
            for pose in self.poses:
                f.write(" ".join(f"{v:.10e}" for v in pose) + "\n")

        # time.txt: one timestamp per line
        time_path = os.path.join(self.seq_dir, "time.txt")
        with open(time_path, "w") as f:
            for ts in self.timestamps:
                f.write(f"{ts:.6f}\n")

        # calib.txt
        calib_path = os.path.join(self.seq_dir, "calib.txt")
        with open(calib_path, "w") as f:
            f.write(format_calib(DEFAULT_P2, DEFAULT_TR))

        # Summary
        duration = 0.0
        if len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0]

        self.get_logger().info("=" * 60)
        self.get_logger().info("Collection complete!")
        self.get_logger().info(f"  Frames:   {self.frame_count}")
        self.get_logger().info(f"  Duration: {duration:.1f}s")
        self.get_logger().info(f"  Output:   {self.seq_dir}")
        self.get_logger().info(f"  Images:   {self.img_dir}")
        self.get_logger().info(f"  Velodyne: {self.vel_dir}")
        self.get_logger().info(f"  Poses:    {poses_path}")
        self.get_logger().info(f"  Calib:    {calib_path}")
        self.get_logger().info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Isaac Sim sensor data to SemanticKITTI format"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "isaac_warehouse"
        ),
        help="Output base directory",
    )
    parser.add_argument("--sequence", type=str, default="01",
                        help="Sequence identifier")
    parser.add_argument("--mode", type=str, default="depth",
                        choices=["depth", "lidar"],
                        help="Sensor mode: depth or lidar")

    # Topic names (defaults match common Isaac Sim ROS2 bridge naming)
    parser.add_argument("--rgb_topic", type=str,
                        default="/front_color_camera/c_rgb/image_raw",
                        help="RGB image topic")
    parser.add_argument("--depth_topic", type=str,
                        default="/front_pdepth_camera/pd/depth",
                        help="Depth image topic (depth mode)")
    parser.add_argument("--lidar_topic", type=str,
                        default="/ouster_lidar/point_cloud",
                        help="LiDAR PointCloud2 topic (lidar mode)")

    # Collection parameters
    parser.add_argument("--save_hz", type=float, default=10.0,
                        help="Target save frequency in Hz")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to collect (0 = unlimited)")
    parser.add_argument("--max_depth", type=float, default=15.0,
                        help="Max depth in meters for filtering")
    parser.add_argument("--depth_subsample", type=int, default=4,
                        help="Subsample depth pixels by this factor (depth mode)")
    parser.add_argument("--use_sim_time", action="store_true",
                        help="Use simulation time from Isaac Sim")

    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    collector = IsaacDataCollector(args)

    if args.use_sim_time:
        collector.set_parameters([
            Parameter("use_sim_time", Parameter.Type.BOOL, True)
        ])

    # Graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        collector.get_logger().info("Ctrl+C received, finalizing...")
        collector.finalize()
        collector.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        pass
    finally:
        if not collector.shutting_down:
            collector.finalize()
        collector.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
