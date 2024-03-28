from typing import Any

import numpy as np

import torch
from torchmetrics import Metric

from ..evaluation import EvalResults
from ..evaluator_base import Sample


class NumGraspsPerObject(Metric):
    def __init__(self,
                 num_unique_objects: int,
                 filter_collisions: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._filter_collisions = filter_collisions
        self._num_unique_objects = num_unique_objects

        self.add_state("num_grasps", default=[], dist_reduce_fx="cat")
        self.add_state("object_id", default=[], dist_reduce_fx="cat")

    def update(self,
               evaluation_results: EvalResults,
               sample: Sample,
               **kwargs: Any) -> None:
        scene_id = sample.scene_id
        eval_per_object = evaluation_results.data.groupby('object_id')
        for object_id, eval_data in eval_per_object:
            if self._filter_collisions:
                eval_data = eval_data.query('cf_suppressed == False and collision == False')

            self.num_grasps.append(torch.tensor(len(eval_data)))
            self.object_id.append(torch.tensor((scene_id << 16) + object_id))

    def compute(self, **kwargs: Any) -> float:
        unique_object_ids = np.unique(self.object_id)
        num_grasps = np.zeros(self._num_unique_objects)
        ng = np.array(self.num_grasps)
        obj_ids = np.array(self.object_id)
        # import pdb; pdb.set_trace()
        for i, object_id in enumerate(unique_object_ids):
            num_grasps[i] = ng[obj_ids == object_id].mean()

        return num_grasps.mean()  # type: ignore
