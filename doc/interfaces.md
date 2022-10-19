# Interfaces
For integration of datasets this package defines interfaces.
Through these interfaces a dataset such as Cornell oder GraspNet can be transformed into a common format such that a training pipeline can easily work on all datasets.

The class [DatasetInterface](../nicr_grasping/dataset/interfaces/interface_base.py) defines a base class which an interface has to inherit from.
It also describes how to implement the abstract methods for a new dataset.