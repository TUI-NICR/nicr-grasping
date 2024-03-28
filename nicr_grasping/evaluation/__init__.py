from dataclasses import dataclass, field
from typing import List, Union, Any, Optional, TYPE_CHECKING

import json

import numpy as np
import pandas as pd

# from ..datatypes.grasp import GraspList
from .. import logger as baselogger

logger = baselogger.getChild('evaluation')

if TYPE_CHECKING:
    from ..datatypes.grasp import GraspList


@dataclass
class EvalParameters:
    top_k: int = 5
    friction_coefficients: List[float] = field(default_factory=list)

    # NMS parameters
    nms_translation_threshold: float = 0.03  # min distance between grasps in meters
    nms_rotation_threshold: float = 30    # min rotation between grasps in degree

    def __post_init__(self) -> None:
        # validate if friction_coefficients is a list
        # this is necessary to save the parameters as json
        if isinstance(self.friction_coefficients, np.ndarray):
            self.friction_coefficients = self.friction_coefficients.tolist()

        if not isinstance(self.friction_coefficients, list):
            raise TypeError("friction_coefficients must be a list")

    def save(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def from_file(cls, filepath: str) -> 'EvalParameters':
        with open(filepath, 'r') as f:
            params = json.load(f)

            return cls(**params)


class EvalResults:
    def __init__(self, grasps: Optional['GraspList'] = None,
                 params: EvalParameters = EvalParameters()) -> None:
        self._params = params

        if grasps is None:
            self._df = pd.DataFrame()
            return

        self._df = pd.DataFrame({
            'grasp': grasps.copy()
        })

        self._df['confidence'] = self._df.apply(lambda row: row.grasp.quality, axis=1, result_type='reduce')
        self._df['object_id'] = self._df.apply(lambda row: row.grasp.object_id, axis=1, result_type='reduce')

        # sort by confidence as most evaluation needs this order
        # e.g. when applying NMS
        self._df.sort_values(by='confidence', ascending=False, inplace=True)

    @property
    def data(self) -> pd.DataFrame:
        return self._df

    @classmethod
    def from_csv(cls, filepath: str) -> 'EvalResults':
        df = pd.read_csv(filepath, index_col=0)

        # load params
        params = EvalParameters.from_file(filepath + '.params')

        obj = cls(params=params)
        obj._df = df

        return obj

    def save(self, path: str) -> None:
        non_object_columns = [c for c, t in self._df.dtypes.items() if t != 'object']

        object_columns = [c for c, t in self._df.dtypes.items() if t == 'object']
        logger.info(f'Ignoring columns {object_columns} when saving evaluation results!')

        self._df.to_csv(path, columns=non_object_columns)
        self._params.save(path + '.params')

    def add_info(self, key: str, default: Any, overwrite: bool = False) -> None:
        if not isinstance(default, (list, np.ndarray)):
            default = [default] * len(self._df)
        assert len(default) == len(self._df), "Info must be of same length as grasps"

        if key in self._df.columns and not overwrite:
            raise ValueError(f"Key {key} already exists")

        self._df[key] = default

    def get_info(self, key: str, original_order: bool = True) -> Any:
        if key not in self._df.columns:
            raise ValueError(f"Key {key} does not exist")

        if original_order:
            return self._df.sort_index()[key]
        else:
            return self._df[key]

    def get_info_for_grasp(self, grasp_index: int) -> Any:
        return self._df.loc[grasp_index]

    def update_info_of_grasp(self, key: str, grasp_index: int, value: Any) -> None:
        if key not in self._df.columns:
            raise ValueError(f"Key {key} does not exist")

        key_index = self._df.columns.get_loc(key)
        self._df.iat[grasp_index, key_index] = value

    def __str__(self) -> str:
        res = ''
        res += 'Grasp evaluation results:\n'
        res += f'Number of grasps: {len(self._df)}\n'
        res += f'Number of grasps after NMS: {len(self._df.query("suppressed == False"))}\n'
        res += str(self._df)
        res += '\n'

        ap_without_nms = self.compute_ap(self._df)
        ap_with_nms = self.compute_ap(self._df.query('suppressed == False'))
        res += f'\nAP (top_k={self._params.top_k}):\n'
        # res += '\t' + str(ap_with_nms.mean())
        res += f'\t{ap_with_nms.mean():.4f} (NMS)\n'
        res += f'\t{ap_without_nms.mean():.4f} (no NMS)\n'

        # collision filtered
        ap_collision_filtered = self.compute_ap(
            self._df.query('cf_suppressed == False and collision == False')
        )
        ap_collision_filtered_no_nms = self.compute_ap(
            self._df.query('collision == False')
        )
        res += f'\nAP (top_k={self._params.top_k}) collision filtered:\n'
        res += f'\t{ap_collision_filtered.mean():.4f} (NMS)\n'
        res += f'\t{ap_collision_filtered_no_nms.mean():.4f} (no NMS)\n'

        return res

    def compute_ap(self, dataframe: Optional[pd.DataFrame] = None, top_k: Optional[int] = None, collision_filtered: bool = False) -> np.ndarray:
        if dataframe is None:
            dataframe = self._df.query('suppressed == False')

        if top_k is None:
            top_k = self._params.top_k

        ap_array = np.zeros((top_k, len(self._params.friction_coefficients)))

        # sort grasps by confidence and replace min_friction with -1 for colliding grasps
        confidence_sorted_df = dataframe.sort_values(by='confidence', ascending=False).copy()

        if collision_filtered:
            confidence_sorted_df = confidence_sorted_df.query('collision == False')

        # as we are evaluating every grasp wether it is in collision or not we need to
        # set the score for colliding grasps to -1 as we would otherwise count them
        # as valid grasps
        confidence_sorted_df.loc[confidence_sorted_df.collision, 'min_friction'] = -1

        score_list = confidence_sorted_df.min_friction.values

        # fill ap_array
        # code taken from graspnetAPI
        for fric_idx, fric in enumerate(np.round(self._params.friction_coefficients, 1)):
            for k in range(0, top_k):
                if k+1 > len(score_list):
                    ap_array[k, fric_idx] = np.sum(((score_list <= fric) & (score_list > 0)).astype(int))/(k+1)
                else:
                    ap_array[k, fric_idx] = np.sum(((score_list[0:k+1] <= fric) & (score_list[0:k+1] > 0)).astype(int))/(k+1)

        return ap_array

    def __add__(self, eval_results: 'EvalResults') -> 'EvalResults':
        assert self._params == eval_results._params, "EvalResults must have same parameters"

        self._df = pd.concat([self._df, eval_results._df], ignore_index=True)

        return self
