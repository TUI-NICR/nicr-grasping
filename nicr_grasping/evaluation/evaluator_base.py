from typing import Optional, Any, Tuple, Union, List
from pathlib import Path
from collections import namedtuple

import time
import enum

from datetime import datetime

import multiprocessing as mp
import functools

import tqdm
import json

import numpy as np

import torch
from torchmetrics import MetricCollection

from . import EvalParameters, EvalResults
from ..datatypes.objects import Scene
from ..datatypes.grasp import ParallelGripperGrasp3DList
from .evaluation import eval_grasps_on_model

from . import logger as baselogger
logger = baselogger.getChild('evaluator')


class Status(enum.Enum):
    SUCCESS = 0
    ERROR = 1
    NO_GRASPS = 2
    SKIPPED = 3


Sample = namedtuple('Sample', ['scene_id', 'ann_id'])


class Evaluator:
    def __init__(self,
                 root_dir: Union[Path, str],
                 params: EvalParameters,
                 split: str,
                 save_dir: Union[Path, str, None] = None,
                 num_tasks_per_child: int = 50,
                 chunksize: int = 50,
                 **kwargs: Any):
        self._root_dir = Path(root_dir)

        assert self._root_dir.exists(), f'Root directory {self._root_dir} does not exist.'

        self._split = split
        self._params = params
        self._save_dir = Path(save_dir) if save_dir is not None else None
        self._samples = self.compute_samples(**kwargs)

        self._num_tasks_per_child = num_tasks_per_child
        self._chunksize = chunksize

        logger.info(f'Loaded {len(self._samples)} samples.')

    @property
    def samples(self) -> List[Sample]:
        return self._samples

    def compute_samples(self, **kwargs: Any) -> List[Sample]:
        """This function should compute all samples the evaluator has to iterate over.
        A sample is a tuple with arbitrary information (e.g. (scene_id, ann_id) for GraspNet-1Billion).

        Returns
        -------
        List[Tuple]
            List of samples.
        """
        raise NotImplementedError()

    @functools.cache
    def create_scene(self, sample: Sample) -> Scene:
        """This function needs to create the necessary scene object
        to evaluate the grasps for the given sample.

        Parameters
        ----------
        sample : Tuple
            Single tuple taken from the result of compute_samples.

        Returns
        -------
        Scene
            Scene object used for evaluation of grasps.
        """
        raise NotImplementedError()

    def load_grasps(self, sample: Sample, **kwargs: Any) -> ParallelGripperGrasp3DList:
        raise NotImplementedError()

    def save_evaluation(self, sample: Sample, evaluation_result: EvalResults, grasps: ParallelGripperGrasp3DList, **kwargs: Any) -> None:
        raise NotImplementedError()

    def load_evaluation(self, sample: Sample, **kwargs: Any) -> Tuple[EvalResults, ParallelGripperGrasp3DList]:
        raise NotImplementedError()

    def can_skip_sample(self, sample: Sample) -> bool:
        raise NotImplementedError()

    def evaluate(self,
                 num_workers: int = 1,
                 metrics: Optional[MetricCollection] = None,
                 **kwargs: Any) -> None:
        res = []

        metafile_name = 'metadata_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
        if self._save_dir is not None:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = self._save_dir / metafile_name
        else:
            metadata_path = Path('/tmp') / metafile_name

        if self._save_dir is not None:
            # save some basic information about evaluation
            # e.g. the source directory containing the grasps which were evaluated
            with open(self._save_dir / 'eval_info.json', 'w') as f:
                json.dump({
                    'source_dir': str(self._root_dir)
                }, f, indent=4)

        logger.info(f'Running evaluation on {len(self._samples)} samples with {num_workers} worker.')

        with metadata_path.open('w') as f:
            f.write('sample,status,runtime\n')
            if num_workers <= 1:
                for sample in tqdm.tqdm(self._samples):
                    job_metadata, eval_result = self._evaluate_sample(sample, **kwargs)
                    f.write(f'"{job_metadata["sample"]}",{job_metadata["status"].name},{job_metadata["runtime"]}\n')

                    if eval_result is not None and metrics is not None:
                        metrics.update(eval_result, sample=job_metadata['sample'])

                    res.append(job_metadata)
            else:
                with mp.Pool(num_workers, maxtasksperchild=self._num_tasks_per_child) as pool:
                    for job_metadata, eval_result in tqdm.tqdm(pool.imap(self._evaluate_sample, self.samples, chunksize=self._chunksize),
                                                               total=len(self.samples)):
                        f.write(f'"{job_metadata["sample"]._asdict()}",{job_metadata["status"].name},{job_metadata["runtime"]}\n')

                        if eval_result is not None and metrics is not None:
                            metrics.update(eval_result, sample=job_metadata['sample'])
                        res.append(job_metadata)

        runtimes = np.array([r['runtime'] for r in res])

        if metrics is not None:
            metric_results = metrics.compute()
            print('Metrics:')
            print(metric_results)

            if self._save_dir is not None:
                # NOTE: we dont need to sync metrics here as we only have one instance
                torch.save(metrics.state_dict(), self._save_dir / f'metrics_{self._split}_topk_{self._params.top_k}.pt')
                with (self._save_dir / f'metrics_{self._split}_topk_{self._params.top_k}.json').open('w') as f:
                    json.dump(metric_results, f, indent=4)

        print('Runtime: %.2f +- %.2f' % (runtimes.mean(), runtimes.std()))

    def _evaluate_sample(self, sample: Sample, **kwargs: Any) -> Tuple[dict, Optional[EvalResults]]:
        start_time = time.time()

        metadata = {
            'sample': sample,
            'runtime': 0,
            'status': Status.ERROR,
        }

        # check if we can skip the sample
        # conditions need to be implemented in the subclass but should generally check if
        # the evaluation results can be loaded
        if self.can_skip_sample(sample):
            results, grasps = self.load_evaluation(sample, **kwargs)
            metadata['status'] = Status.SKIPPED

            if kwargs.get('vis', False):
                # need to load scene here for visualization but in general we dont want
                # to load the scene if we can skip the sample as creating a scene can be
                # expensive
                scene = self.create_scene(sample)

                scene.show(grasps,
                           eval_results=results)
            return metadata, results

        scene = self.create_scene(sample)

        try:
            grasps = self.load_grasps(sample)
        except Exception as e:
            logger.error(f'Failed to load grasps for sample {sample}: {e}')
            runtime = time.time() - start_time

            metadata['runtime'] = runtime
            return metadata, None

        unassigned_grasps = list(filter(lambda g: g.object_id is None, grasps))
        if len(unassigned_grasps) != 0 and len(unassigned_grasps) != len(grasps):
            raise ValueError('Only part of the grasps are not assigned to objects!')
        elif len(unassigned_grasps) == len(grasps):
            scene.assign_grasps_to_objects(grasps)

        # sort grasps by model
        # evaluate grasps on each model
        # add model identifier to results
        results = EvalResults(grasps, self._params)
        results.add_info('contact', False)
        results.add_info('contact_points', None)
        results.add_info('min_friction', np.nan)
        results.add_info('collision', False)
        results.add_info('suppressed', False)
        results.add_info('cf_suppressed', False)

        for key, default in scene._collision_checker.INFO_KEYS.items():
            results.add_info(key, default)

        for obj_id, df in results.data.groupby('object_id'):
            grasps_for_obj = ParallelGripperGrasp3DList(
                df.grasp.tolist()
            )
            collision_res, collision_infos = scene.check_collision(grasps_for_obj,
                                                                   object_id=obj_id,
                                                                   **kwargs)

            res = eval_grasps_on_model(grasps_for_obj,
                                       scene.objects[obj_id],
                                       self._params)

            res.add_info('collision', collision_res)

            for key, default in scene._collision_checker.INFO_KEYS.items():
                res.add_info(key, default)

            # compute nms results
            # NOTE: we can do this after evaluation of grasp qualities, because
            #       we evaluate every grasp independently of NMS results
            suppressed = grasps_for_obj.nms(self._params.nms_translation_threshold,
                                            self._params.nms_rotation_threshold)

            collision_filtered_grasps = ParallelGripperGrasp3DList(
                [g for i, g in enumerate(grasps_for_obj) if not collision_res[i]]
            )
            cf_suppressed = collision_filtered_grasps.nms(
                self._params.nms_translation_threshold,
                self._params.nms_rotation_threshold
            )
            collision_filtered_supressed = np.zeros(len(grasps_for_obj), dtype=bool)
            collision_filtered_supressed[~collision_res] = cf_suppressed

            res.add_info('suppressed', suppressed)
            res.add_info('cf_suppressed', collision_filtered_supressed)

            for gi in range(len(grasps_for_obj)):
                for key in scene._collision_checker.INFO_KEYS.keys():
                    res.update_info_of_grasp(key, gi, collision_infos[gi][key])

            # to keep the indices of the dataframe entries consistent we
            # overwrite the index and use them to update our evaluation results
            res.data.index = df.index
            results.data.loc[df.index] = res.data

        results.add_info('global_suppressed', grasps.nms(self._params.nms_translation_threshold,
                                                         self._params.nms_rotation_threshold))
        # compute nms for collision filtered grasps
        # this is done on the whole scene
        g = ParallelGripperGrasp3DList(
            [r.grasp for i, r in results.data.iterrows() if not r.collision]
        )
        suppressed = np.zeros(len(grasps), dtype=bool)
        suppressed[~results.data.collision] = g.nms(self._params.nms_translation_threshold,
                                                    self._params.nms_rotation_threshold)
        results.add_info('global_cf_suppressed', suppressed)

        if kwargs.get('vis', False):
            scene.show(grasps,
                       eval_results=results)

        if self._save_dir is not None:
            self.save_evaluation(sample, results, grasps, **kwargs)

        runtime = time.time() - start_time

        # need to drop contact_points as they are not pickleable
        # which results in errors when using multiprocessing
        results.data.drop(columns=['contact_points'], inplace=True)

        metadata['runtime'] = runtime
        metadata['status'] = Status.SUCCESS

        return metadata, results
