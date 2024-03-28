from typing import Any, List, Union

import numpy as np
import torch
from torchmetrics import Metric

from ..evaluation import EvalResults
from ..evaluator_base import Sample


class APPerFrame(Metric):
    """
    Computes AP for whole frame meaning there is no distinction between objects.
    Metric can reach maximum value even if only one of many objects has assigned grasps.
    """
    def __init__(self,
                 top_k: Union[int, List[int]] = [50],
                 filter_collisions: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(top_k, list):
            top_k = [top_k]

        self._top_k = top_k
        self._filter_collisions = filter_collisions

        self.add_state("ap", default=[], dist_reduce_fx="cat")
        self.add_state("top_k", default=[], dist_reduce_fx="cat")

    def update(self,
               evaluation_results: EvalResults,
               **kwargs: Any) -> None:
        if self._filter_collisions:
            eval_data = evaluation_results.data.query('global_cf_suppressed == False and collision == False')
        else:
            eval_data = evaluation_results.data.query('global_suppressed == False')

        max_top_k = max(self._top_k)
        ap_array = evaluation_results.compute_ap(eval_data, top_k=max_top_k)

        for top_k in self._top_k:
            ap = ap_array[:top_k].mean()
            self.ap.append(torch.tensor(ap))
            self.top_k.append(torch.tensor(top_k))

    def compute(self, **kwargs: Any) -> float:
        res = {}
        top_ks = np.array(self.top_k)
        aps = np.array(self.ap)
        for top_k in self._top_k:
            ap = aps[top_ks == top_k].mean()
            res[top_k] = ap
        return res  # type: ignore


class APPerObject(Metric):
    def __init__(self,
                 num_unique_objects: int,
                 num_samples_per_object: int,
                 top_k: Union[int, List[int]] = [5],
                 filter_collisions: bool = False,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not isinstance(top_k, list):
            top_k = [top_k]

        self._top_k = top_k
        self._filter_collisions = filter_collisions
        self._num_unique_objects = num_unique_objects
        self._num_samples_per_object = num_samples_per_object

        self.add_state("ap", default=[], dist_reduce_fx="cat")
        self.add_state("object_id", default=[], dist_reduce_fx="cat")
        self.add_state("top_k", default=[], dist_reduce_fx="cat")

    def update(self,
               evaluation_results: EvalResults,
               sample: Sample,
               **kwargs: Any) -> None:
        scene_id = sample.scene_id
        eval_per_object = evaluation_results.data.groupby('object_id')
        for object_id, eval_data in eval_per_object:
            if self._filter_collisions:
                eval_data = eval_data.query('cf_suppressed == False and collision == False')
            else:
                eval_data = eval_data.query('suppressed == False')

            max_top_k = max(self._top_k)
            ap_array = evaluation_results.compute_ap(eval_data, top_k=max_top_k)

            for top_k in self._top_k:
                ap = ap_array[:top_k].mean()

                self.ap.append(torch.tensor(ap))
                self.object_id.append(torch.tensor((scene_id << 16) + object_id))
                self.top_k.append(torch.tensor(top_k))

    def compute(self, **kwargs: Any) -> float:
        unique_object_ids = np.unique(self.object_id)
        ap = np.array(self.ap)
        obj_ids = np.array(self.object_id)
        top_ks = np.array(self.top_k)
        # import pdb; pdb.set_trace()

        res = {}

        for top_k in self._top_k:
            aps = np.zeros(self._num_unique_objects)
            ap_top_k = ap[top_ks == top_k]
            obj_ids_top_k = obj_ids[top_ks == top_k]
            for i, object_id in enumerate(unique_object_ids):
                aps[i] = ap_top_k[obj_ids_top_k == object_id].sum() / self._num_samples_per_object

            res[top_k] = aps.mean()

        return res  # type: ignore
