import os
from setuptools import setup, find_packages


def run_setup():
    # get version
    version_namespace = {}
    with open(os.path.join('nicr_grasping', 'version.py')) as version_file:
        exec(version_file.read(), version_namespace)
    version = version_namespace['_get_version'](with_suffix=False)

    # setup
    setup(name='nicr_grasping',
          version='{}.{}.{}'.format(*version),
          description='Package for core datatypes and dataset handling of grasping datasets.',
          author='Benedict Stephan',
          author_email='benedict.stephan@tu-ilmenau.de',
          license=('Copyright 2017, Neuroinformatics and Cognitive Robotics '
                   'Lab TU Ilmenau, Ilmenau, Germany'),
          install_requires=[
              'numpy',
              'tqdm',
              'shapely',
              'rtree',
              'imageio',
              'pytest-benchmark',
              'opencv-python',
              'scikit-image',
              'matplotlib',
              'torch',
              'pandas',
              'autolab_core',
              'trimesh',
              'open3d',
              'cvxopt'
          ],
          packages=find_packages())


if __name__ == '__main__':
    run_setup()
