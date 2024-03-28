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

### IROS2022 - On the Importance of Label Encoding and Uncertainty Estimation for Robotic Grasp Detection
For more details see [here](projects/grasp_encodings_and_uncertainties/iros2022_graspnet_uncertainties.md).

### (Submitted) CASE2024 - GraspTrack: Object and Grasp Pose Tracking for Arbitrary Objects
For more details see [here](projects/grasptrack/grasp_pose_tracking.md).

# Changelog
**Version 0.2.0 (Mar 27, 2024)**
- reworked evaluation pipeline to be more modular
- added visualization for evaluation pipeline
- fixed typehints

**Version 0.1.0 (Oct 5, 2023)**
- added evaluation pipeline
- added external dependency to dexnet and meshpy (code is directly included as those packages are not actively maintained and not python3 compatible).
- added GraspTrack project readme
- added testcases