import socket
import os
from pathlib import Path

GRASPNET_PATH = Path('SET_PATH_HERE')

def graspnet_dataset_path():
    if 'GRASPNET_PATH' in os.environ:
        path = Path(os.environ['GRASPNET_PATH'])
    else:
        path = GRASPNET_PATH

    if not path.exists():
        raise FileNotFoundError('Graspnet dataset does not exists for this device. Searched in ' + str(path))

    return path
