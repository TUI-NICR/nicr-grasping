import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

from ..datatypes.grasp import GraspList
from ..datatypes.objects.graspable_objects import ObjectModel
from ..datatypes.objects.collision_objects import CollisionObject
from ..collision.pointcloud_checker import PointCloudChecker

from . import EvalParameters


GRASPS_VISIBLE = True
CURRENT_VIS_MODE = 'mesh'

class Scene:

    STATIC_COLLISION_OBJECT_ID_OFFSET = 1000

    def __init__(self,
                 collision_checker = PointCloudChecker()):
        self._objects = []
        self._static_collision_objects = []
        self._collision_scene_needs_update = True
        self._collision_scene = None

        print('Using collision checker:', collision_checker.__class__.__name__)
        self._collision_checker = collision_checker

    def add_object(self, obj: ObjectModel):
        self._objects.append(obj)
        self._collision_scene_needs_update = True

    def add_static_collision_object(self, obj: CollisionObject):
        self._static_collision_objects.append(obj)
        self._collision_scene_needs_update = True

    @property
    def collision_scene(self):
        if not self._collision_scene_needs_update:
            return self._collision_scene
        else:
            self._compute_collision_scene()
            return self._collision_scene

    def check_collision(self, grasps: GraspList,
                        **kwargs):
        if self._collision_scene_needs_update:
            self._compute_collision_scene()

        collision_infos = []
        is_in_collision = []

        for grasp in grasps:
            collision = self._collision_checker.check_collision(grasp, **kwargs)
            is_in_collision.append(collision)
            collision_infos.append(self._collision_checker.collision_info)

        return is_in_collision, collision_infos

    def _compute_collision_scene(self):
        point_clouds = []
        point_labels = []

        # collect samples pointcloud from all objects
        for oi, obj in enumerate(self._objects):
            obj_points = obj.sample_points()
            obj_points = np.concatenate([obj_points, np.ones((len(obj_points), 1))], axis=-1)
            obj_points = obj.pose.transformation_matrix @ obj_points.T
            obj_points = obj_points.T[:, :3]
            point_clouds.append(obj_points)
            point_labels.append(np.ones((len(obj_points), 1)) * oi)

        for oi, obj in enumerate(self._static_collision_objects):
            object_points = obj.sample_points()
            point_clouds.append(object_points)
            point_labels.append(np.ones((len(object_points), 1)) * oi + self.STATIC_COLLISION_OBJECT_ID_OFFSET)

        self._collision_scene = np.concatenate(point_clouds, axis=0)
        point_labels = np.concatenate(point_labels, axis=0)

        # set point cloud in collision checker
        self._collision_checker.set_point_cloud(self._collision_scene,
                                                labels=point_labels)
        self._collision_scene_needs_update = False

    def eval_grasps(self, grasps: GraspList, eval_parameters: EvalParameters = None):
        # sort grasps by model
        # evaluate grasps on each model
        # add model identifier to results
        pass

    def show(self, grasps=None, simple_grasps=False):
        # plot scene with all objects
        cmap = plt.get_cmap('Set1')

        scene_meshs = []
        scene_pcs = []
        for oi, obj in enumerate(self._objects):

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(obj.mesh.vertices)
            mesh.triangles = o3d.utility.Vector3iVector(obj.mesh.triangles)
            mesh.compute_vertex_normals()

            mesh = mesh.transform(obj.pose.transformation_matrix)

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(obj.sample_points())
            pc = pc.transform(obj.pose.transformation_matrix)

            pc.paint_uniform_color(cmap(oi)[:3])
            mesh.paint_uniform_color(cmap(oi)[:3])

            scene_meshs.append(mesh)
            scene_pcs.append(pc)

        static_collision_objects = []
        for oi, obj in enumerate(self._static_collision_objects):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(obj.point_cloud)

            pc.paint_uniform_color(cmap(oi)[:3])

            static_collision_objects.append(pc)

        viz = o3d.visualization.VisualizerWithKeyCallback()
        viz.create_window()

        for mesh in scene_meshs:
            viz.add_geometry(mesh)

        for coll_obj in static_collision_objects:
            viz.add_geometry(coll_obj)

        grasp_meshs = []
        if grasps is not None:
            for grasp in grasps:
                g_mesh = grasp.open3d_geometry(simple_grasps)
                g_mesh.transform(grasp.transformation_matrix)

                viz.add_geometry(g_mesh)
                grasp_meshs.append(g_mesh)

        def _toggle_mode(viz):
            global CURRENT_VIS_MODE
            if CURRENT_VIS_MODE == 'mesh':
                # toggle to collision scene
                for mesh in scene_meshs:
                    viz.remove_geometry(mesh, False)
                for pc in scene_pcs:
                    viz.add_geometry(pc, False)
                CURRENT_VIS_MODE = 'collision'
            elif CURRENT_VIS_MODE == 'collision':
                # toggle to mesh scene
                for mesh in scene_meshs:
                    viz.add_geometry(mesh, False)
                for pc in scene_pcs:
                    viz.remove_geometry(pc, False)
                CURRENT_VIS_MODE = 'mesh'
            else:
                raise RuntimeError
            return False

        def _toggle_grasps(viz):
            global GRASPS_VISIBLE
            if GRASPS_VISIBLE:
                # remove grasps from scene
                for g_mesh in grasp_meshs:
                    viz.remove_geometry(g_mesh, False)
            else:
                # add grasps to scene
                for g_mesh in grasp_meshs:
                    viz.add_geometry(g_mesh, False)
            GRASPS_VISIBLE = not GRASPS_VISIBLE

            # viz._vis_grasps_visible = not viz._vis_grasps_visible


        viz.register_key_callback(ord('C'), _toggle_mode)

        viz.register_key_callback(ord('G'), _toggle_grasps)

        viz.run()

    def show_collision_scene(self, grasps=None):
        # plot scene with all objects
        cmap = plt.get_cmap('Set1')

        scene_objects = []
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.collision_scene)

        scene_objects.append(pc)

        if grasps is not None:
            for grasp in grasps:
                g_mesh = grasp.open3d_geometry()
                g_mesh.transform(grasp.transformation_matrix)

                scene_objects.append(g_mesh)

        viz = o3d.visualization.VisualizerWithKeyCallback()
        viz.create_window()

        for scene_object in scene_objects:
            viz.add_geometry(scene_object)

        viz.run()
