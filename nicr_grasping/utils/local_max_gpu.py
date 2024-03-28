import numpy as np
from typing import Union

import torch
from torch.nn import MaxPool2d
from torch.nn import functional as F
import math

CUDA_AVAILABLE = torch.cuda.is_available()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


def peak_local_max_2d(img: Union[np.ndarray, torch.Tensor],
                      min_distance: int = 10,
                      threshold_abs: float = 0.1,
                      num_peaks: int = 5, device: Union[torch.device, str] = "cpu") -> np.ndarray:
    if not CUDA_AVAILABLE:
        print("No GPU availalbe!")
        device = "cpu"
    # tensor from image, add batch and channel dimension
    if not isinstance(img, torch.Tensor):
        img_gpu = torch.from_numpy(img[np.newaxis, ...]).to(device)
    else:
        img_gpu = img.to(device)
        # we need 3 dims for maxpooling
        if img_gpu.dim() < 3:
            img_gpu = img_gpu.unsqueeze(0)

    # check data range 0.0...1.0
    # assert img_gpu.min() >= 0.0 and img_gpu.max() <= 1.0, print("peak_local_max: img not in range [0.0,1.0]")

    # init MaxPool2d layer
    kernel_size = 2*min_distance
    # assert kernel_size <= img_gpu.shape[2] and kernel_size <= img_gpu.shape[3]
    pool = MaxPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0, return_indices=True)

    # padding
    py = img_gpu.shape[1] % kernel_size
    px = img_gpu.shape[2] % kernel_size
    img_gpu = F.pad(img_gpu, (0, px, 0, py), value=-1.0)

    # apply pooling
    img_gpu_pooled, img_indices = pool(img_gpu)

    img_indices = np.unravel_index(torch.squeeze(img_indices).cpu().numpy(), img_gpu.shape[-2:])
    peak_values = img_gpu[0, img_indices[0], img_indices[1]]
    # print(peak_values)
    # print(peak_values.shape)
    # sort
    sorted_indices = torch.argsort(peak_values.view(-1), descending=True)[:num_peaks].cpu().numpy()
    sorted_values = peak_values.view(-1)[sorted_indices]
    # threshold
    threshold_mask = (sorted_values >= threshold_abs).cpu().numpy()
    sorted_indices = sorted_indices[threshold_mask]

    if len(sorted_indices) == 0:
        return np.array([])
    else:
        img_indices = np.array([img_indices[0].flatten()[sorted_indices], img_indices[1].flatten()[sorted_indices]])
        img_indices = img_indices.T
        return img_indices
