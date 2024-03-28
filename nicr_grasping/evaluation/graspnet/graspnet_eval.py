from typing import Dict, List, Any

import numpy as np

from graspnetAPI import GraspNetEval

import open3d as o3d

from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.eval_utils import eval_grasp, transform_points
from graspnetAPI.utils.utils import generate_scene_model


class GraspNetSceneEval(GraspNetEval):
    def __init__(self, root: str, camera: str = 'kinect', split: str = 'test') -> None:
        super().__init__(root, camera, split)

    def eval_grasp_group_for_scene(self,
                                   scene_id: int,
                                   TOP_K: int,
                                   vis: bool,
                                   max_width: float,
                                   log_dict: Dict, config: Dict,
                                   table: np.ndarray, do_extensive_eval: bool,
                                   list_coe_of_friction: List[float], dexmodel_list: List[Any],
                                   model_sampled_list: List[np.ndarray], scene_accuracy: Dict, ann_id: int, grasp_group: GraspGroup) -> Dict:
        _, pose_list, camera_pose, align_mat = self.get_model_poses(
            scene_id, ann_id)
        table_trans = transform_points(
            table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

        gg_array = grasp_group.grasp_group_array
        min_width_mask = (gg_array[:, 1] < 0)
        max_width_mask = (gg_array[:, 1] > max_width)
        gg_array[min_width_mask, 1] = 0
        gg_array[max_width_mask, 1] = max_width
        grasp_group.grasp_group_array = gg_array

        eval_res = eval_grasp(
            grasp_group, model_sampled_list, dexmodel_list, pose_list, config,
            table=table_trans, voxel_size=0.008, TOP_K=None if do_extensive_eval else TOP_K, fill_seperated_masks=log_dict is not None)

        for key, e_res in eval_res.items():
            grasp_list = e_res['grasp_list']
            score_list = e_res['score_list']
            collision_mask_list = e_res['collision_mask_list']
            empty_list = e_res['empty_list']
            sort_idx_list = e_res['sort_idx_list']
            seperated_collision_mask_list = e_res['seperated_collision_mask_list']
            object_ids = e_res['object_ids']

            # remove empty
            grasp_list = [x for x in grasp_list if len(x) != 0]
            score_list = [x for x in score_list if len(x) != 0]
            object_ids = [x for x in object_ids if len(x) != 0]
            if log_dict is not None:
                seperated_collision_mask_list = [
                    seperated_collision_mask_list[idx] for idx, x in enumerate(collision_mask_list) if len(x) != 0
                ]
            collision_mask_list = [
                x for x in collision_mask_list if len(x) != 0]

            if len(grasp_list) == 0:
                # TODO: fix logging
                # if log_dict: log_dict["empty_grasp_list_count"] += 1
                grasp_accuracy = np.zeros((TOP_K, len(list_coe_of_friction)))
                scene_accuracy[key].append(grasp_accuracy)
                scene_accuracy[key + '_cf'].append(grasp_accuracy)

                print('\rMean Accuracy for scene:{} ann:{}='.format(
                    scene_id, ann_id), np.mean(grasp_accuracy[:, :]))

                return scene_accuracy

                # concat into scene level
            grasp_list, score_list, collision_mask_list = np.concatenate(
                grasp_list), np.concatenate(score_list), np.concatenate(collision_mask_list)
            object_ids = np.concatenate(object_ids)

            if log_dict is not None:
                left_list = []
                right_list = []
                bottom_list = []
                inner_list = []
                for x in seperated_collision_mask_list:
                    left, right, bottom, inner = x
                    left_list.append(left)
                    right_list.append(right)
                    bottom_list.append(bottom)
                    inner_list.append(inner)
                seperated_collision_mask_list = [
                    np.concatenate(left_list) & collision_mask_list,
                    np.concatenate(right_list) & collision_mask_list,
                    np.concatenate(bottom_list) & collision_mask_list,
                    np.concatenate(inner_list) & collision_mask_list,
                ]

                log_dict[key]["unfiltered_grasp_counts"].append(
                    len(grasp_list))
                log_dict[key]["collision_counts"]["left"].append(
                    int(np.sum(~seperated_collision_mask_list[0])))
                log_dict[key]["collision_counts"]["right"].append(
                    int(np.sum(~seperated_collision_mask_list[1])))
                log_dict[key]["collision_counts"]["bottom"].append(
                    int(np.sum(~seperated_collision_mask_list[2])))
                log_dict[key]["collision_counts"]["inner"].append(
                    int(np.sum(seperated_collision_mask_list[3])))

                log_dict[key]["collision_counts"]["combined"].append(
                    int(np.sum(~collision_mask_list)))

                log_dict[key]["scores"]["unfiltered"].append(
                    score_list.tolist())
                log_dict[key]["confidences"]["unfiltered"].append(
                    grasp_list[:, 0].tolist())
                log_dict[key]["scores"]["rejected"].append(
                    score_list[collision_mask_list].tolist())
                log_dict[key]["confidences"]["rejected"].append(
                    grasp_list[collision_mask_list][:, 0].tolist())
                log_dict[key]["object_ids"].append(object_ids.tolist())

            if vis:
                t = o3d.geometry.PointCloud()
                t.points = o3d.utility.Vector3dVector(table_trans)
                model_list = generate_scene_model(
                    self.root, 'scene_%04d' % scene_id, ann_id, return_poses=False, align=False, camera=self.camera)
                import copy
                gg = GraspGroup(copy.deepcopy(grasp_list))
                scores = np.array(score_list)
                scores = scores / 4 + 0.75  # -1 -> 0, 0 -> 0.5, 1 -> 1
                scores[collision_mask_list] = 0
                gg.scores = scores
                gg.widths = 0.1 * np.ones((len(gg)), dtype=np.float32)
                grasps_geometry = gg.to_open3d_geometry_list()
                pcd = self.loadScenePointCloud(scene_id, self.camera, ann_id)

                camera_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()

                global_axis = o3d.geometry.TriangleMesh.create_coordinate_frame()
                global_axis.transform(np.linalg.inv(np.matmul(align_mat, camera_pose)))

                o3d.visualization.draw_geometries([pcd, *grasps_geometry,
                                                   camera_axis, global_axis])
                o3d.visualization.draw_geometries(
                    [pcd, *grasps_geometry, *model_list])
                o3d.visualization.draw_geometries(
                    [*grasps_geometry, *model_list, t])

            if log_dict is not None:
                log_dict[key]["scores"]["after_collision"].append(
                    score_list.tolist())
                log_dict[key]["confidences"]["after_collision"].append(
                    grasp_list[:, 0].tolist())

                log_dict[key]["collision_lists"].append(
                    collision_mask_list.tolist())
                log_dict[key]["score_lists"].append(score_list.tolist())
                log_dict[key]["sortidx_lists"].append(sort_idx_list)

                # sort in scene level
            grasp_confidence = grasp_list[:, 0]
            indices = np.argsort(-grasp_confidence)
            grasp_list, score_list, collision_mask_list = grasp_list[
                indices], score_list[indices], collision_mask_list[indices]

            if do_extensive_eval:
                TOP_K = 50

                # calculate AP
            grasp_accuracy = self.calculate_ap(
                TOP_K, list_coe_of_friction, score_list)

            print(key, '\tMean Accuracy for scene:%04d ann:%04d = %.3f' % (
                scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:, :])), flush=True)
            scene_accuracy[key].append(grasp_accuracy)
            if log_dict is not None:
                log_dict[key]["grasp_accuracies"].append(
                    grasp_accuracy.tolist())

            grasp_list = grasp_list[~collision_mask_list]
            score_list = score_list[~collision_mask_list]
            collision_mask_list = collision_mask_list[~collision_mask_list]
            grasp_accuracy = self.calculate_ap(
                TOP_K, list_coe_of_friction, score_list)
            scene_accuracy[key+'_cf'].append(grasp_accuracy)
            print(key, 'CF', '\tMean Accuracy for scene:%04d ann:%04d = %.3f' % (
                scene_id, ann_id, 100.0 * np.mean(grasp_accuracy[:, :])), flush=True)
        return scene_accuracy
