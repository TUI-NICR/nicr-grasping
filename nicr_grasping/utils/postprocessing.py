from typing import List
import torch
import numpy as np

from skimage.feature import peak_local_max as peak_local_max_skimage
from ..utils.local_max_gpu import peak_local_max_2d as peak_local_max_torch

from ..datatypes.grasp import RectangleGrasp, RectangleGraspList

def peak_local_max( quality_map : torch.Tensor,
                    num_peaks   : int = -1,
                    min_distance : int = 20,
                    threshold_abs  : float = 0.2,
                    loc_max_version : str = "skimage_cpu") -> np.ndarray:

    if loc_max_version == "skimage_cpu":
        if not isinstance(quality_map, np.ndarray):
            quality_map_cpu = quality_map.cpu().numpy()
        else:
            quality_map_cpu = quality_map
        local_max = peak_local_max_skimage(
            quality_map_cpu, min_distance=min_distance, threshold_abs=threshold_abs, num_peaks=num_peaks)

    # using 2*min_distance works best for the torch implementation
    # this prevents cluttered grasps (no explicit min_distance check is done for better performance)
    elif loc_max_version == "torch_gpu":
        local_max = peak_local_max_torch(
            quality_map, min_distance=2*min_distance, threshold_abs=threshold_abs, num_peaks=num_peaks, device="cuda:0")

    elif loc_max_version == "torch_cpu":
        local_max = peak_local_max_torch(
            quality_map, min_distance=2*min_distance, threshold_abs=threshold_abs, num_peaks=num_peaks, device="cpu")

    else:
        raise NotImplementedError

    return local_max

def convert_ggcnn_output_to_grasps(model_output : List[np.ndarray],
                                   num_grasps   : int = -1,
                                   min_distance : int = 20,
                                   min_quality  : float = 0.2,
                                   loc_max_version : str = "skimage_cpu") -> List[RectangleGrasp]:
    # invert angle
    if len(model_output) == 3:
        quality_map, angle_map, width_map = model_output
    elif len(model_output) == 4:
        quality_map, cos_map, sin_map, width_map = model_output
        angle_map = np.arctan2(sin_map, cos_map)

    angle_map *= -1
    cos_map = np.cos(angle_map)
    sin_map = np.sin(angle_map)
    # width_map *= 150

    return convert_model_output_to_grasps([quality_map, cos_map, sin_map, width_map],
                                          num_grasps=num_grasps,
                                          min_distance=min_distance,
                                          min_quality=min_quality,
                                          loc_max_version=loc_max_version)



def convert_model_output_to_grasps(model_output : List[np.ndarray],
                                   num_grasps   : int = -1,
                                   min_distance : int = 20,
                                   min_quality  : float = 0.2,
                                   loc_max_version : str = "skimage_cpu",
                                   additional_maps = None) -> List[RectangleGrasp]:
    if len(model_output) == 3:
        quality_map, angle_map, width_map = model_output
    elif len(model_output) == 4:
        quality_map, cos_map, sin_map, width_map = model_output
        angle_map = np.arctan2(sin_map, cos_map)

    quality_map = quality_map.squeeze()
    angle_map = angle_map.squeeze()
    width_map = width_map.squeeze()

    grasps = []

    if num_grasps != -1:

        local_max = peak_local_max(
            quality_map, min_distance=min_distance, threshold_abs=min_quality, num_peaks=num_grasps, loc_max_version=loc_max_version)

        grasps = []
        for grasp_point_array in local_max:
            grasp_point = tuple(grasp_point_array)

            grasp_angle = angle_map[grasp_point]
            quality = quality_map[grasp_point]
            width = width_map[grasp_point]

            if additional_maps is not None:
                additional_params = {key: additional_maps[key].squeeze()[grasp_point].cpu().numpy() for key in additional_maps}
            else:
                additional_params = None

            # bring center in correct format
            # [[x, y]]
            center = np.array(grasp_point)[::-1]
            center = center.reshape(-1, 2)


            # TODO: Creation of RectangleGrasp calls _compute_points which could be done on GPU. This would require a constructor for GraspLists which takes multiple sets of points.
            if isinstance(quality_map, torch.Tensor):
                g = RectangleGrasp(quality=quality.cpu().numpy(), center=center, angle=grasp_angle.cpu().numpy(), width=width.cpu().numpy(),
                                   additional_params=additional_params)
            else:
                g = RectangleGrasp(quality=quality, center=center, angle=grasp_angle, width=width,
                                   additional_params=additional_params)

            grasps.append(g)
    else:
        # grasp_positions = cv2.findNonZero(quality_map)
        quality_map[quality_map < min_quality] = 0
        grasp_positions = np.array(np.nonzero(quality_map)).transpose()

        grasps = []
        for grasp_point_array in grasp_positions:
            grasp_point = tuple(grasp_point_array)

            grasp_angle = angle_map[grasp_point]
            quality = quality_map[grasp_point]

            # bring center in correct format
            # [[x, y]]
            center = np.array(grasp_point)[::-1]
            center = center.reshape(-1, 2)

            if width_map is not None:
                width = width_map[grasp_point]
                g = RectangleGrasp(quality=quality, center=center, angle=grasp_angle, width=width)
            else:
                g = RectangleGrasp(quality=quality, center=center, angle=grasp_angle)

            grasps.append(g)

    # sort grasps with highest quality one at the front
    grasps = RectangleGraspList(grasps)
    grasps.sort()

    return grasps
