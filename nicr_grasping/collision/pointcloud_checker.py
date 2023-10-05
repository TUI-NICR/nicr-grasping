import numpy as np

from .collision_checker import CollisionChecker
from ..datatypes.grasp import ParallelGripperGrasp3D


def pointcloud_from_depthimage(depth_image, intrinsics, anchor=np.zeros((1, 2)), scaling=np.ones((1, 2))):
    num_valid_points = (depth_image > 0).sum()

    pc = np.zeros((num_valid_points, 3))

    height, width, _ = depth_image.shape

    u, v = np.meshgrid(
        np.arange(width, dtype=float),
        np.arange(height, dtype=float)
    )

    # scale the pixel coordinates
    u *= scaling[0, 0]
    v *= scaling[0, 1]

    # shift the pixel coordinates by the anchor
    u += anchor[0, 0]
    v += anchor[0, 1]

    pz = depth_image[:, :, 0]
    px = pz * (u - intrinsics.cx) / intrinsics.fx
    py = pz * (v - intrinsics.cy) / intrinsics.fy

    p = np.stack([px, py, pz], axis=-1)
    pc = p.reshape(-1, 3)
    pc = pc[pc[:, 2] > 0]

    return pc


class PointCloudChecker(CollisionChecker):
    def __init__(self, intrinsics=None):
        self.intrinsics = intrinsics
        self.point_cloud = None
        self._point_labels = None

        self._collision_info = None
        self._collision_info_is_valid = False

    @property
    def collision_info(self):
        if self._collision_info_is_valid:
            return self._collision_info
        else:
            return None

    def set_depth_image(self, depth_image, anchor=np.zeros((1, 2)), scaling=np.ones((1, 2))):
        if self.intrinsics is None:
            raise ValueError("intrinsics must be set before setting a depth image")

        pc = pointcloud_from_depthimage(depth_image, self.intrinsics, anchor, scaling)

        # save pc as homogeneous coordinates as we will transform it with a 4x4 matrix
        self.point_cloud = np.concatenate([pc, np.ones((len(pc), 1))], axis=-1)

        self._collision_info_is_valid = False

    def set_point_cloud(self, point_cloud, labels=None):
        pc_shape = point_cloud.shape

        assert len(pc_shape) == 2

        if labels is not None:
            assert len(labels) == len(point_cloud), f"labels must have the same length as point cloud: {len(labels)} != {len(point_cloud)}"
            self._point_labels = labels

        if pc_shape[1] == 3:
            # convert to homogeneous coordinates
            self.point_cloud = np.concatenate([point_cloud, np.ones((pc_shape[0], 1))], axis=-1)
        elif pc_shape[1] == 4:
            self.point_cloud = point_cloud
        else:
            raise ValueError("point cloud must have either 3 or 4 columns")

        self._collision_info_is_valid = False

    def check_collision(self, grasp: ParallelGripperGrasp3D, **kwargs):

        assert isinstance(grasp, ParallelGripperGrasp3D)

        # check is done with the following steps
        # 1. crop pointcloud around grasp pose to reduce points for collision checking
        #       we do this with a large margin to not remove to many points
        # 2. transform the point cloud to the grasp frame
        # 3. check if any point is inside the gripper by thresholding the pointcloud

        grasp_position = grasp.position
        min_corner = grasp_position - 0.1
        max_corner = grasp_position + 0.1

        cropped_point_cloud = self.point_cloud[
            np.logical_and(self.point_cloud[:, :3] >= min_corner, self.point_cloud[:, :3] <= max_corner).all(axis=1)
        ]

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
        # TODO: picture of what the gipper looks like in the grasp frame

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
