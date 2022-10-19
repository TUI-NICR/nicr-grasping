import scipy.sparse as sp

from nicr_grasping.dataset.interfaces.cornell_interface import CornellInterface

CORNELL_ROOT = '/datasets_nas/grasping/cornell'

def test_cornell_load(tmpdir):
    dest_dir = tmpdir / 'cornell_test_load'
    dataset_interface = CornellInterface(CORNELL_ROOT,
                                         dest_dir)

    dataset_interface.convert(num_samples=1)

    # load sample labels
    quality_label = sp.load_npz(str(dest_dir / 'cornell' / 'grasp_labels' / '000_quality.npz'))
    assert quality_label.shape == (480, 640)
    assert quality_label.max() == 1

    assert len(dataset_interface) == 885

    for split in ['train', 'test']:

        assert (dest_dir / dataset_interface.DATASET_NAME / f'{split}.json').exists()
