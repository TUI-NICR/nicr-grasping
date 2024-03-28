import numpy as np
from typing import Dict, Any, Optional

from .pointcloud_checker import PointCloudChecker
from ..datatypes.grasp import ParallelGripperGrasp3D, Grasp3D
from ..datatypes.intrinsics import PinholeCameraIntrinsic


class GraspNetChecker(PointCloudChecker):

    INFO_KEYS = {
        **PointCloudChecker.INFO_KEYS,
        'is_empty': False,
        'num_inner_points': 0
    }

    def __init__(self, intrinsics: Optional[PinholeCameraIntrinsic] = None) -> None:
        super().__init__(intrinsics)

    def check_collision(self,
                        grasp: Grasp3D,
                        frames: Dict[str, np.ndarray] = {},
                        object_id: int = 0,
                        **kwargs: Any) -> bool:
        # IMPORTANT: this checker should not be used because it can result in false negatives
        #            as the pointcloud is cropped around the object and not the gripper

        assert isinstance(grasp, ParallelGripperGrasp3D)
        assert self.point_cloud is not None, "GraspNetChecker requires pointcloud"

        world_to_camera = frames.get('camera', None)
        assert world_to_camera is not None, "GraspnetChecker requires camera frame"

        camera_to_world = np.linalg.inv(world_to_camera)

        # checking collision the same way as done in graspnetAPI
        # 1. crop the pointcloud around the associated object
        #   this is a problematic step as points around the gripper might be cropped out
        #   leading to false negatives
        # 2. transform pointcloud into grasp pose
        # 3. check collisions by counting point within gripper

        # transform pointcloud into camera frame and determine bbox of object
        collision_pointcloud_in_cam = (camera_to_world @ self.point_cloud.T).T
        if self._point_labels is None:
            object_pc = collision_pointcloud_in_cam
        else:
            object_pc = collision_pointcloud_in_cam[self._point_labels.reshape(-1) == object_id]

        min_corner = object_pc[:, :3].min(axis=0) - 0.05
        max_corner = object_pc[:, :3].max(axis=0) + 0.05

        # crop pointcloud and transform back into world frame
        cropped_point_cloud = collision_pointcloud_in_cam[
            np.logical_and(collision_pointcloud_in_cam[:, :3] > min_corner, collision_pointcloud_in_cam[:, :3] < max_corner).all(axis=1)
        ]

        # transform pointcloud
        # NOTE: as we normally compute collisions in world frame the grasp poses
        #       are in world frame as well so we need to transform them into camera frame
        # NOTE: the following thresholds differ in their applied dimension as our definition
        #       of the gripper differs from the graspnetAPI definition (see conversion code)
        grasp_pose = camera_to_world @ grasp.transformation_matrix

        transformed_point_cloud = (np.linalg.inv(grasp_pose) @ cropped_point_cloud.T).T

        # crop pointcloud to the gripper bounding box

        # parameters taken from graspnetAPI
        # NOTE: naming convention is as follows:
        #       - extent in x direction: width
        #       - extent in y direction: height
        #       - extent in z direction: depth

        gripper_depth_base = grasp.gripper_parameters.base_depth
        gripper_finger_width = grasp.gripper_parameters.finger_width
        gripper_finger_depth = grasp.gripper_parameters.finger_depth

        gripper_base_offset = grasp.gripper_parameters.base_offset

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
        collision_base = (pc[:, 2] <= (-gripper_base_offset))
        num_collisions_base = collision_base.sum()

        # do we have collision with the left finger?
        # NOTE: as the orientation of the gripper differs from our definition
        #       and we have rotated the gripper (see conversion) left and right have swapped meaning
        collision_right = (pc[:, 0] <= (min_x + gripper_finger_width))
        collision_right[collision_base] = False
        num_collisions_right = collision_right.sum()

        # do we have collision with the right finger?
        collision_left = (pc[:, 0] >= (max_x - gripper_finger_width))
        collision_left[collision_base] = False
        num_collisions_left = collision_left.sum()

        # check if enough points are between the fingers
        # graspnetAPI counts grasps withouth enough points between the fingers as in collision
        is_between_fingers = (pc[:, 0] > (min_x + gripper_finger_width)) & \
            (pc[:, 0] < (max_x - gripper_finger_width)) &\
            (pc[:, 2] > min_z + gripper_depth_base)

        is_between_fingers = np.logical_and(is_between_fingers, ~collision_base)

        num_between_fingers = is_between_fingers.sum()
        is_empty = num_between_fingers < kwargs.get('empty_thresh', 10)

        collision_mask_right = np.zeros(len(self.point_cloud), dtype=bool)
        collision_mask_right[pc_indices[collision_right]] = True

        collision_mask_left = np.zeros(len(self.point_cloud), dtype=bool)
        collision_mask_left[pc_indices[collision_left]] = True

        collision_mask_base = np.zeros(len(self.point_cloud), dtype=bool)
        collision_mask_base[pc_indices[collision_base]] = True

        self._collision_info = {
            'collision_base': int(num_collisions_base),
            'collision_left': int(num_collisions_left),
            'collision_right': int(num_collisions_right),
            'is_empty': is_empty,
            'num_inner_points': num_between_fingers,
            'collision_mask_right': collision_mask_right,
            'collision_mask_left': collision_mask_left,
            'collision_mask_base': collision_mask_base,
        }
        self._collision_info_is_valid = True

        return bool(num_collisions_base or num_collisions_left or num_collisions_right or is_empty)
