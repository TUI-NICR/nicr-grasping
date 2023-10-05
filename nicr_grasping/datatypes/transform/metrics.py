import enum

import numpy as np

from .pose import Pose


class DifferenceType(enum.Enum):
    EUCLIDEAN = 0
    ROTATION = 1
    DEG5CM5 = 2


class SymmetryType(enum.Enum):
    NONE = 0
    Y = 1
    Y_FLIP = 2
    FLIP = 3


def _diff_euclidean(pose1: Pose, pose2: Pose, **kwargs):
    res = {
        'error': np.linalg.norm(pose1.position - pose2.position)
    }
    return res


def _diff_deg5cm5(pose1: Pose, pose2: Pose, **kwargs):
    rot = _diff_rotation(pose1, pose2)
    trans = _diff_euclidean(pose1, pose2)
    if rot['error'] < 5 and trans['error'] < 0.05:
        return {
            'positive': True
        }
    else:
        return {
            'positive': False
        }


def _diff_rotation(pose1: Pose, pose2: Pose, symmetry: SymmetryType = SymmetryType.NONE, **kwargs) -> dict:
    if symmetry == SymmetryType.NONE:
        trans_diff = pose1.inverse().transformation_matrix @ pose2.transformation_matrix
        rot_diff_matrix = trans_diff[:3, :3]
        # compute angle axis representation
        trace = rot_diff_matrix.trace()
        if trace > 3:
            trace = 3
        elif trace < -1:
            trace = -1
        cos_angle = (rot_diff_matrix.trace() - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)

    elif symmetry == SymmetryType.Y:
        # to compute difference of only roll and pitch we rotate the y axis with both rotations
        # and compute the angly between the results
        y = np.array([0, 1, 0])
        y1 = pose1.transformation_matrix[:3, :3] @ y
        y2 = pose2.transformation_matrix[:3, :3] @ y
        scalar_product = np.dot(y1, y2) / (np.linalg.norm(y1) * np.linalg.norm(y2))
        scalar_product = np.clip(scalar_product, -1, 1)
        angle = np.arccos(scalar_product)

    elif symmetry == SymmetryType.Y_FLIP:
        y_flip_rotation = np.diag([-1.0, 1.0, -1.0])
        y_flip_transformation = np.eye(4)
        y_flip_transformation[:3, :3] = y_flip_rotation
        pose1_flipped = pose1.transform(y_flip_transformation)

        error = _diff_rotation(pose1, pose2, symmetry=SymmetryType.NONE)
        error_flipped = _diff_rotation(pose1_flipped, pose2, symmetry=SymmetryType.NONE)

        # we directly return the error as we already have the correct output
        if error['error'] < error_flipped['error']:
            return error
        else:
            return error_flipped

    elif symmetry == SymmetryType.FLIP:
        x_flip_rotation = np.diag([1.0, -1.0, -1.0])
        y_flip_rotation = np.diag([-1.0, 1.0, -1.0])
        z_flip_rotation = np.diag([-1.0, -1.0, 1.0])

        x_flip_transformation = np.eye(4)
        y_flip_transformation = np.eye(4)
        z_flip_transformation = np.eye(4)

        x_flip_transformation[:3, :3] = x_flip_rotation
        y_flip_transformation[:3, :3] = y_flip_rotation
        z_flip_transformation[:3, :3] = z_flip_rotation

        pose1_x_flipped = pose1.transform(x_flip_transformation)
        pose1_y_flipped = pose1.transform(y_flip_transformation)
        pose1_z_flipped = pose1.transform(z_flip_transformation)

        error = _diff_rotation(pose1, pose2, symmetry=SymmetryType.NONE)

        error_x_flipped = _diff_rotation(pose1_x_flipped, pose2, symmetry=SymmetryType.NONE)
        error_y_flipped = _diff_rotation(pose1_y_flipped, pose2, symmetry=SymmetryType.NONE)
        error_z_flipped = _diff_rotation(pose1_z_flipped, pose2, symmetry=SymmetryType.NONE)

        # we directly return the error as we already have the correct output
        return {
            'error': min(error['error'], error_x_flipped['error'], error_y_flipped['error'], error_z_flipped['error'])
        }
    else:
        raise ValueError('Unknown symmetry type: ' + symmetry.name)

    angle = np.rad2deg(angle)

    res = {
        'error': angle
    }

    return res


def difference(pose1: Pose, pose2: Pose,
               difference_type: DifferenceType = DifferenceType.EUCLIDEAN, **kwargs) -> dict:
    if difference_type == DifferenceType.EUCLIDEAN:
        return _diff_euclidean(pose1, pose2, **kwargs)
    elif difference_type == DifferenceType.DEG5CM5:
        return _diff_deg5cm5(pose1, pose2, **kwargs)
    elif difference_type == DifferenceType.ROTATION:
        return _diff_rotation(pose1, pose2, **kwargs)
