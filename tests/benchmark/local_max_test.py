import torch
import pytest
import numpy as np
from nicr_grasping.utils.postprocessing import peak_local_max
import copy

@pytest.mark.parametrize("loc_max_version", ["skimage_cpu", "torch_cpu", "torch_gpu"])
@pytest.mark.benchmark(group='peak_local_max')
def test_local_max_benchmark(benchmark, loc_max_version):
    quality_map = np.random.rand(320, 320)
    quality_map = torch.from_numpy(quality_map)
    min_distance = 10
    min_quality = 0.1
    num_grasps = 10

    benchmark(  peak_local_max, copy.deepcopy(quality_map),
                min_distance=min_distance, threshold_abs=min_quality,
                num_peaks=num_grasps, loc_max_version=loc_max_version)