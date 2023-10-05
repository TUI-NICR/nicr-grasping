import os
import os.path as osp
import sys

from typing import Optional

import numpy as np
import open3d as o3d

from . import SceneObject
from ..transform import Pose

from ...external.meshpy import ObjFile, SdfFile, Mesh3D

from ...external.dexnet.grasping.graspable_object import GraspableObject3D

MODEL_CACHE_DIR = '/tmp/model_cache'

class ObjectModel(SceneObject, GraspableObject3D):
    def __init__(self, mesh: ObjFile, sdf: Optional[SdfFile]=None, key='', model_name='', mass=1.0, convex_pieces=None):
        GraspableObject3D.__init__(self, sdf, mesh, key, model_name, mass, convex_pieces)
        SceneObject.__init__(self)
        self.pose = Pose()
        self._sampled_points = None

    @classmethod
    def from_dir(cls, directory: str, model_name: str='textured'):
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
            vertices = np.load(osp.join(cache_path, 'vertices.npy'))
            faces = np.load(osp.join(cache_path, 'faces.npy'))
            normals = np.load(osp.join(cache_path, 'normals.npy'))

            mesh = Mesh3D(vertices, faces, normals)
        else:
            # no cache so we load from .obj and save to cache
            mesh = ObjFile(osp.join(directory, f'{model_name}.obj')).read()

            os.makedirs(cache_path)

            np.save(osp.join(cache_path, 'vertices.npy'), mesh.vertices)
            np.save(osp.join(cache_path, 'faces.npy'), mesh.triangles)
            np.save(osp.join(cache_path, 'normals.npy'), mesh.normals)

        sdf = SdfFile(osp.join(directory, f'{model_name}.sdf')).read()

        obj = cls(mesh, sdf)

        # check if we have a ply of sampled points
        ply_path = osp.join(directory, f'nontextured.ply')
        if osp.exists(ply_path):
            pc = o3d.io.read_point_cloud(ply_path)
            pc = pc.voxel_down_sample(0.008)
            obj._sampled_points = pc.points

        return obj

    def sample_points(self, num_samples: int = 500):
        if self._sampled_points:
            return self._sampled_points

        vertex_indices = np.random.choice(self.mesh.vertices.shape[0], num_samples)
        return self.mesh.vertices[vertex_indices]
