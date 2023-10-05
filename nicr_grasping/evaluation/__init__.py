from dataclasses import dataclass
from typing import List, Union, Any, Optional

import json

import numpy as np
import pandas as pd

from ..datatypes.grasp import GraspList

@dataclass
class EvalParameters:
    top_k: int
    friction_coefficients: List[float]

    def save(self, filepath=str):
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'r') as f:
            params = json.load(f)

            return cls(**params)


class EvalResults:
    def __init__(self, grasps: Optional[GraspList] = None, params: Optional[EvalParameters] = None):
        self._grasps = grasps
        self._df = None
        self._params = params

        if grasps is not None:
            self.add_info('confidence', np.zeros(len(grasps)))
            self.add_info('gripper_width', np.zeros(len(grasps)))
            self.add_info('gripper_height', np.zeros(len(grasps)))

            for gi, grasp in enumerate(grasps):
                self.update_info_of_grasp('confidence', gi, grasp.quality)
                self.update_info_of_grasp('gripper_width', gi, grasp.width)
                self.update_info_of_grasp('gripper_height', gi, grasp.height)

    @classmethod
    def from_csv(cls, filepath: str):
        df = pd.read_csv(filepath, index_col=0)

        # load params
        params = EvalParameters.from_file(filepath + '.params')

        obj = cls(params=params)
        obj._df = df

        return obj

    def save(self, path: str):
        self._df.to_csv(path)
        self._params.save(path + '.params')

    def add_info(self, key: str, default: Union[List, np.ndarray], overwrite: bool = False):
        if not isinstance(default, (list, np.ndarray)):
            default = [default] * len(self._grasps)
        assert len(default) == len(self._grasps), "Info must be of same length as grasps"

        if self._df is not None:
            if key in self._df.columns and not overwrite:
                raise ValueError(f"Key {key} already exists")
        else:
            self._df = pd.DataFrame(index=range(len(self._grasps)), columns=[key])

        self._df[key] = default

    def get_info(self, key: str):
        if key not in self._df.columns:
            raise ValueError(f"Key {key} does not exist")
        return self._df[key]

    def update_info_of_grasp(self, key: str, grasp_index: int, value: Any):
        if key not in self._df.columns:
            raise ValueError(f"Key {key} does not exist")

        self._df.loc[grasp_index, key] = value

    def __str__(self) -> str:
        res = ''
        res += 'Grasp evaluation results:\n'
        res += f'Number of grasps: {len(self._grasps)}\n'
        res += str(self._df)
        res += '\n'

        ap = self.compute_ap()
        res += f'AP (top_k={self._params.top_k}):\n'
        res += str(ap.mean())

        return res

    def compute_ap(self, top_k=None, collision_filtered=False):
        if top_k is None:
            top_k = self._params.top_k

        ap_array = np.zeros((top_k, len(self._params.friction_coefficients)))

        # sort grasps by confidence and replace min_friction with -1 for colliding grasps
        confidence_sorted_df = self._df.sort_values(by='confidence', ascending=False).copy()

        if collision_filtered:
            confidence_sorted_df = confidence_sorted_df.query('collision == False')
        else:
            confidence_sorted_df.loc[confidence_sorted_df.collision, 'min_friction'] = -1

        score_list = confidence_sorted_df.min_friction.values

        # fill ap_array
        # code taken from graspnetAPI
        for fric_idx, fric in enumerate(self._params.friction_coefficients):
            for k in range(0,top_k):
                if k+1 > len(score_list):
                    ap_array[k,fric_idx] = np.sum(((score_list<=fric) & (score_list>0)).astype(int))/(k+1)
                else:
                    ap_array[k,fric_idx] = np.sum(((score_list[0:k+1]<=fric) & (score_list[0:k+1]>0)).astype(int))/(k+1)

        return ap_array

    def __add__(self, eval_results: 'EvalResults'):
        assert self._params == eval_results._params, "EvalResults must have same parameters"

        self._df = pd.concat([self._df, eval_results._df], ignore_index=True)
        self._grasps.extend(eval_results._grasps)

        return self
