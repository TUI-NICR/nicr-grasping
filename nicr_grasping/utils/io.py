import numpy as np

from scipy.spatial.transform import Rotation as R

def mira_json_to_pose(json_object: dict):

    position = np.array([
        json_object['X'],
        json_object['Y'],
        json_object['Z']
    ])

    orientation_x = R.from_euler('x', json_object['Roll'], degrees=True)
    orientation_y = R.from_euler('y', json_object['Pitch'], degrees=True)
    orientation_z = R.from_euler('z', json_object['Yaw'], degrees=True)

    orientation_matrix = orientation_z.as_matrix() @ orientation_y.as_matrix() @ orientation_x.as_matrix()

    return position, orientation_matrix
