# NICR-Grasping

This package contains a general definition of a structure for grasp estimation datasets.
For detailed listing of documentation see [here](doc/overview.md).

# Datasets

## Graspnet-1Billion
To use this repository in conjunction with the Graspnet-1Billion dataset the following steps are required:
1. Install the `graspnetapi` package from [our fork](https://github.com/best3125/graspnetAPI.git).
This should only be necessary if you want to use this repository for evaluation.
2. Set the `GRASPNET_PATH` environment variable or set the path directly in `nicr_grasping.utils.paths.py` so that your local dataset can be found by this repository.

# Publications

This repository was used for the following publications

## IROS2022 - On the Importance of Label Encoding and Uncertainty Estimation for Robotic Grasp Detection
For more details see [here](projects/iros2022_graspnet_uncertainties.md).
