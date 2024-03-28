import os
import os.path as osp
import sys

from typing import Optional, Union, List

import numpy as np

from .. import logger

from . import SceneObject
from ..transform import Pose

from ...external.meshpy import ObjFile, SdfFile, Mesh3D, Sdf3D

from ...external.dexnet.grasping.graspable_object import GraspableObject3D

MODEL_CACHE_DIR = os.environ.get('NICR_GRASPING_MODEL_CACHE_DIR', '/tmp/model_cache')


class ObjectModel(SceneObject, GraspableObject3D):
    def __init__(self,
                 mesh: Mesh3D, sdf: Optional[Sdf3D] = None,
                 key: str = '', model_name: str = '', mass: float = 1.0, convex_pieces: Union[List[Mesh3D], None] = None) -> None:
        GraspableObject3D.__init__(self, sdf, mesh, key, model_name, mass, convex_pieces)
        SceneObject.__init__(self, pose=Pose())
        self._sampled_points: Union[np.ndarray, None] = None

    @classmethod
    def from_dir(cls,
                 directory: str, model_name: str = 'textured', use_presampled_collision_points: bool = True) -> 'ObjectModel':
        """Load ObjectModel from directory containing necessary files.
        It is asumed that the following files exist within the directory:
        - {model_name}.obj: mesh file
        - {model_name}.sdf: sdf file

        If not already cached, the mesh files are loaded and cached in MODEL_CACHE_DIR.

        Parameters
        ----------
        directory : str
            directory containing the files to be loaded
        model_name : str, optional
            optional name of the model. this is used for finding the .obj and .sdf files, by default 'textured'

        Returns
        -------
        ObjectModel
            _description_
        """
        assert osp.exists(directory), f"Directory {directory} does not exist"

        # check if we have a cached version of this model
        absolute_path = osp.abspath(directory)
        model_id = absolute_path.replace('/', '_')

        cache_path = osp.join(MODEL_CACHE_DIR, model_id)

        if osp.exists(cache_path):
            # cache exists so we can load the Mesh3D from numpy files
            logger.debug(f'Loading cached version of {model_name} from {cache_path}')

            vertices = np.load(osp.join(cache_path, 'vertices.npy'))
            faces = np.load(osp.join(cache_path, 'faces.npy'))
            normals = np.load(osp.join(cache_path, 'normals.npy'))

            mesh = Mesh3D(vertices, faces, normals)

            sdf = Sdf3D(
                np.load(osp.join(cache_path, 'sdf_data.npy')),
                np.load(osp.join(cache_path, 'sdf_origin.npy')),
                np.load(osp.join(cache_path, 'sdf_resolution.npy'))
            )
        else:
            # no cache so we load from .obj and save to cache
            logger.debug(f'No model cache found for {model_name} in {cache_path}')

            mesh = ObjFile(osp.join(directory, f'{model_name}.obj')).read()

            os.makedirs(cache_path)

            np.save(osp.join(cache_path, 'vertices.npy'), mesh.vertices)
            np.save(osp.join(cache_path, 'faces.npy'), mesh.triangles)
            np.save(osp.join(cache_path, 'normals.npy'), mesh.normals)

            sdf = SdfFile(osp.join(directory, f'{model_name}.sdf')).read()

            np.save(osp.join(cache_path, 'sdf_data.npy'), sdf.data_)
            np.save(osp.join(cache_path, 'sdf_origin.npy'), sdf.origin_)
            np.save(osp.join(cache_path, 'sdf_resolution.npy'), sdf.resolution_)

        obj = cls(mesh, sdf)

        presampled_collision_points_path = osp.join(cache_path, f'presampled_collision_points.npy')
        if use_presampled_collision_points and os.path.exists(presampled_collision_points_path):
            presampled_points = np.load(presampled_collision_points_path)
            obj._sampled_points = presampled_points
        else:
            import open3d as o3d
            # check if we have a ply of sampled points
            ply_path = osp.join(directory, f'nontextured.ply')
            if osp.exists(ply_path):
                pc = o3d.io.read_point_cloud(ply_path)
                pc = pc.voxel_down_sample(0.008)
                obj._sampled_points = np.array(pc.points)

                if not os.path.exists(presampled_collision_points_path):
                    np.save(presampled_collision_points_path, obj._sampled_points)

        return obj

    def sample_points(self, num_samples: int = 500) -> np.ndarray:
        if self._sampled_points is not None:
            return self._sampled_points

        vertex_indices = np.random.choice(self.mesh.vertices.shape[0], num_samples)
        return self.mesh.vertices[vertex_indices]
