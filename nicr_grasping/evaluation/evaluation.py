import os
import os.path as osp
import sys
from typing import List, Union, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import open3d as o3d

from ..datatypes.transform import Pose
from ..datatypes.grasp import GraspList
from ..datatypes.grasp_conversion import CONVERTER_REGISTRY

from ..collision import PointCloudChecker

from ..external.dexnet.grasping.graspable_object import GraspableObject3D
# from ...external.dexnet.
from ..external.meshpy import ObjFile, SdfFile, Mesh3D
from ..external.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from ..external.dexnet.grasping.quality import PointGraspMetrics3D
from ..external.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory

from . import EvalParameters, EvalResults
from .scene import Scene
from ..datatypes.objects import ObjectModel, CollisionObject


MODEL_CACHE_DIR = '/tmp/model_cache'


def get_config():
    '''
     - return the config dict
    '''
    config = dict()
    force_closure = dict()
    force_closure['quality_method'] = 'force_closure'
    force_closure['num_cone_faces'] = 8
    force_closure['soft_fingers'] = 1
    force_closure['quality_type'] = 'quasi_static'
    force_closure['all_contacts_required']= 1
    force_closure['check_approach'] = False
    force_closure['torque_scaling'] = 0.01
    force_closure['wrench_norm_thresh'] = 0.001
    force_closure['wrench_regularizer'] = 0.0000000001
    config['metrics'] = dict()
    config['metrics']['force_closure'] = force_closure
    return config


def eval_grasps_on_model(grasps, model, eval_parameters: EvalParameters) -> EvalResults:
    config = get_config()

    # make copy of grasps as we are transforming them
    grasps_copy = grasps.copy()

    results = EvalResults(grasps_copy, eval_parameters)
    min_friction = np.ones(len(grasps_copy)) * np.nan
    is_in_contact_list = np.zeros(len(grasps_copy)).astype(bool)

    results.add_info('min_friction', min_friction)
    results.add_info('contact', is_in_contact_list)

    # transform grasps from camera frame into object frame
    grasps_copy.transform(model.pose.inverse())

    for grasp_index in range(len(grasps_copy)):
        grasp = grasps_copy[grasp_index]
        dexnet_grasp = CONVERTER_REGISTRY.convert(grasp, ParallelJawPtGrasp3D)

        # compute contact points of grasp on model
        # these can be reused for grasp quality computation
        # as only the friction coefficient changes
        is_in_contact, contacts = dexnet_grasp.close_fingers(
            model,
            check_approach=config['metrics']['force_closure']['check_approach']
        )

        # if grasp does not result in contact we can skip it
        # otherwise update list
        if not is_in_contact:
            continue
        results.update_info_of_grasp('contact', grasp_index, True)

        # iterate over friction coefficients
        # starting with the hightes one
        # as soon as a friction coefficient does not result in a force closure grasp
        # we can stop
        # IMPORTANT: we round the scores to 4 decimal points to avoid numerical issues as they are compared to used friction coefficients
        friction_coefficients = np.round(eval_parameters.friction_coefficients, 4)

        for friction in np.sort(friction_coefficients)[::-1]:

            # TODO: this needs a better solution
            config['metrics']['force_closure']['friction_coef'] = friction
            dexnet_config = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])

            is_force_closure = PointGraspMetrics3D.grasp_quality(
                dexnet_grasp,
                model,
                dexnet_config,
                contacts=contacts)

            if is_force_closure:
                results.update_info_of_grasp('min_friction', grasp_index, friction)
            else:
                break

    return results
