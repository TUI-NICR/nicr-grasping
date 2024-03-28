import os

from pathlib import Path

GRASPNET_VAR_NAME = 'NICR_GRASPING_GRASPNET_PATH'


def graspnet_dataset_path() -> Path:
    graspnet_path = os.environ.get(GRASPNET_VAR_NAME)
    if graspnet_path is None:
        raise ValueError('Graspnet dataset path not set. Please set the environment variable ' + GRASPNET_VAR_NAME)

    path = Path(graspnet_path)

    if not path.exists():
        raise FileNotFoundError('Graspnet dataset does not exists for this device. Searched in ' + str(path))

    return path
