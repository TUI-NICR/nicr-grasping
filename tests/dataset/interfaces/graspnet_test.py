import pytest
import scipy.sparse as sp

pytest.importorskip('graspnetAPI')

from nicr_grasping.utils.paths import graspnet_dataset_path
from nicr_grasping.dataset.interfaces.graspnet_interface import GraspNetInterface

GRASPNET_ROOT = graspnet_dataset_path()

def test_graspnet_load(tmpdir):

    dest_dir = tmpdir / 'graspnet_test_load'
    dataset_interface = GraspNetInterface(GRASPNET_ROOT,
                                          dest_dir)
    dataset_interface.convert(num_samples=1)

    assert len(dataset_interface) == 48640

    quality_label = sp.load_npz(str(dest_dir / dataset_interface.DATASET_NAME / 'grasp_labels' / '00000_quality.npz'))
    assert quality_label.shape == (720, 1280)

    for split in ['train', 'test', 'test_seen', 'test_similar', 'test_novel']:

        assert (dest_dir / dataset_interface.DATASET_NAME / f'{split}.json').exists()
