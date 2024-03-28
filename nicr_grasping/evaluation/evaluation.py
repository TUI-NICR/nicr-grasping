from typing import Any, Dict

import numpy as np

from ..datatypes.grasp_conversion import CONVERTER_REGISTRY
from ..datatypes.grasp.grasp_lists import ParallelGripperGrasp3DList
from ..datatypes.objects.graspable_objects import ObjectModel

from ..external.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from ..external.dexnet.grasping.quality import PointGraspMetrics3D
from ..external.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory

from . import EvalParameters, EvalResults


# TODO: this should be a dataclass. this would need some rewriting of dexnet code
def get_config() -> dict:
    '''
     - return the config dict
    '''
    config: Dict[str, Any] = dict()
    force_closure: Dict[str, Any] = dict()
    force_closure['quality_method'] = 'force_closure'
    force_closure['num_cone_faces'] = 8
    force_closure['soft_fingers'] = 1
    force_closure['quality_type'] = 'quasi_static'
    force_closure['all_contacts_required'] = 1
    force_closure['check_approach'] = False
    force_closure['torque_scaling'] = 0.01
    force_closure['wrench_norm_thresh'] = 0.001
    force_closure['wrench_regularizer'] = 0.0000000001
    config['metrics'] = dict()
    config['metrics']['force_closure'] = force_closure
    return config


def eval_grasps_on_model(grasps: ParallelGripperGrasp3DList,
                         model: ObjectModel,
                         eval_parameters: EvalParameters) -> EvalResults:
    config = get_config()

    results = EvalResults(grasps, eval_parameters)
    # NOTE: after constructing the results object we only work with the underlying dataframe
    #       through the results.data attribute as this automatically keeps the order of the grasps
    #       through its index. This allows to keep correspondences between the results and the original grasps
    #       even if the order of grasps is changed during evaluation (e.g. by sorting by confidence).

    results.add_info('min_friction', np.nan)
    results.add_info('contact', False)
    results.add_info('contact_points', None)

    for grasp_index in results.data.index:

        grasp = results.data.at[grasp_index, 'grasp']

        grasp.transform(model.pose.inverse())
        dexnet_grasp = CONVERTER_REGISTRY.convert(grasp, ParallelJawPtGrasp3D)

        # compute contact points of grasp on model
        # these can be reused for grasp quality computation
        # as only the friction coefficient changes
        is_in_contact, contacts = dexnet_grasp.close_fingers(
            model,
            check_approach=config['metrics']['force_closure']['check_approach'],
        )

        # if grasp does not result in contact we can skip it
        # otherwise update list
        if not is_in_contact:
            continue
        # results.update_info_of_grasp('contact', grasp_index, True)
        # results.update_info_of_grasp('contact_points', grasp_index, contacts)
        # row.contact = True
        # row.contact_points = contacts
        results.data.at[grasp_index, 'contact'] = True
        results.data.at[grasp_index, 'contact_points'] = contacts

        # iterate over friction coefficients
        # starting with the hightes one
        # as soon as a friction coefficient does not result in a force closure grasp
        # we can stop
        # IMPORTANT: we round the scores to 4 decimal points to avoid numerical issues as they are compared to used friction coefficients
        friction_coefficients = np.round(eval_parameters.friction_coefficients, 4)

        for friction in np.sort(friction_coefficients)[::-1]:

            config['metrics']['force_closure']['friction_coef'] = friction
            dexnet_config = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])

            is_force_closure = PointGraspMetrics3D.grasp_quality(
                dexnet_grasp,
                model,
                dexnet_config,
                contacts=contacts)

            if is_force_closure:
                # results.update_info_of_grasp('min_friction', grasp_index, friction)
                results.data.at[grasp_index, 'min_friction'] = friction
            else:
                break

    return results
