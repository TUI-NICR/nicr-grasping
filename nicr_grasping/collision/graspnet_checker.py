import numpy as np

import open3d as o3d

from .pointcloud_checker import PointCloudChecker
from ..datatypes.grasp import ParallelGripperGrasp3D


class GraspNetChecker(PointCloudChecker):
    def __init__(self, intrinsics=None):
        super().__init__(intrinsics)

    def check_collision(self,
                        grasp: ParallelGripperGrasp3D,
                        object_id: int = 0,
                        world_to_camera: np.ndarray = None,
                        **kwargs):
        # IMPORTANT: this checker should not be used because it can result in false negatives
        #            as the pointcloud is cropped around the object and not the gripper

        assert isinstance(grasp, ParallelGripperGrasp3D)

        assert world_to_camera is not None, "GraspnetChecker requires world_to_camera transformation matrix"

        # checking collision the same way as done in graspnetAPI
        # 1. crop the pointcloud around the associated object
        #   this is a problematic step as points around the gripper might be cropped out
        #   leading to false negatives
        # 2. transform pointcloud into grasp pose
        # 3. check collisions by counting point within gripper
        # TODO: graspnet computes the bbox of the object in camera coordinates
        #       we compute the bbox in world coordinates

        # transform pointcloud into camera frame and determine bbox of object
        collision_pointcloud_in_cam = (world_to_camera @ self.point_cloud.T).T
        object_pc = collision_pointcloud_in_cam[self._point_labels.reshape(-1) == object_id]

        min_corner = object_pc[:, :3].min(axis=0) - 0.05
        max_corner = object_pc[:, :3].max(axis=0) + 0.05

        # crop pointcloud and transform back into world frame
        cropped_point_cloud_in_cam = collision_pointcloud_in_cam[
            np.logical_and(collision_pointcloud_in_cam[:, :3] > min_corner, collision_pointcloud_in_cam[:, :3] < max_corner).all(axis=1)
        ]
        cropped_point_cloud = (np.linalg.inv(world_to_camera) @ cropped_point_cloud_in_cam.T).T

        # transform pointcloud
        grasp_pose = grasp.transformation_matrix
        transformed_point_cloud = (np.linalg.inv(grasp_pose) @ cropped_point_cloud.T).T

        # crop pointcloud to the gripper bounding box

        # parameters taken from graspnetAPI
        # TODO: move these parameters to a config file or class for better parameter management
        # NOTE: naming convention is as follows:
        #       - extent in x direction: width
        #       - extent in y direction: height
        #       - extent in z direction: depth

        gripper_depth_base = 0.01
        gripper_finger_width = 0.01
        gripper_finger_depth = 0.04

        gripper_base_offset = 0.02

        min_z = -gripper_depth_base - gripper_base_offset
        max_z = gripper_finger_depth - gripper_base_offset

        min_x = -grasp.width/2 - gripper_finger_width
        max_x = grasp.width/2 + gripper_finger_width

        min_y = -grasp.height/2
        max_y = grasp.height/2

        # crop
        is_in_x = (transformed_point_cloud[:, 0] >= min_x) & (transformed_point_cloud[:, 0] <= max_x)
        is_in_y = (transformed_point_cloud[:, 1] >= min_y) & (transformed_point_cloud[:, 1] <= max_y)
        is_in_z = (transformed_point_cloud[:, 2] >= min_z) & (transformed_point_cloud[:, 2] <= max_z)

        pc = transformed_point_cloud[(is_in_x & is_in_y & is_in_z)]
        pc_indices = np.arange(len(transformed_point_cloud))[(is_in_x & is_in_y & is_in_z)]

        # do we have contact with the gripper base?
        collision_base = (pc[:, 2] < (gripper_depth_base / 2 - gripper_base_offset))
        num_collisions_base = collision_base.sum()

        # do we have collision with the left finger?
        collision_left = (pc[:, 0] < (min_x + gripper_finger_width))
        num_collisions_left = collision_left.sum()

        # do we have collision with the right finger?
        collision_right = (pc[:, 0] > (max_x - gripper_finger_width))
        num_collisions_right = collision_right.sum()

        self._collision_info = {
            'base': int(num_collisions_base),
            'left': int(num_collisions_left),
            'right': int(num_collisions_right),
            'cropped_pc': pc,
            'base_collision_points': self.point_cloud[pc_indices][collision_base],
            'left_collision_points': self.point_cloud[pc_indices][collision_left],
            'right_collision_points': self.point_cloud[pc_indices][collision_right]
        }
        self._collision_info_is_valid = True

        return bool(num_collisions_base or num_collisions_left or num_collisions_right)
