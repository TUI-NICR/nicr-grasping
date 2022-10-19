import pytest
import json

from nicr_grasping.dataset.split_generation import SplitGenerator


@pytest.fixture
def test_dataset_metadata():
    metadata = {
        "num_total_samples": 2233,
        "modalities": ["color_image", "depth_image"]
    }
    return metadata


@pytest.fixture
def test_dataset(tmp_path, test_dataset_metadata):
    dataset_dir = tmp_path / 'test_dataset'
    dataset_dir.mkdir()


    with (dataset_dir / 'metadata.json').open("w") as f:
        json.dump(test_dataset_metadata, f)

    return dataset_dir


@pytest.fixture
def absolute_split():
    return {'train': 150, 'test': 20}


@pytest.fixture
def percent_split():
    return {'train': 0.8, 'test': 0.2}


@pytest.fixture
def absolute_split_generator(absolute_split, test_dataset):
    return SplitGenerator(test_dataset, split_sizes=absolute_split)


@pytest.fixture
def percent_split_generator(percent_split, test_dataset):
    return SplitGenerator(test_dataset, split_percs=percent_split)


def test_absolute_split_generation(test_dataset, absolute_split_generator, absolute_split, test_dataset_metadata):

    absolute_split_generator.generate_splits()

    assert (test_dataset / 'test.json').exists()
    assert (test_dataset / 'train.json').exists()

    for split, split_size in absolute_split.items():
        with (test_dataset / f'{split}.json').open('r') as f:
            split_data = json.load(f)

            assert len(split_data['input_files']) == split_size
            assert len(split_data['label_files']) == split_size

            # check if modalities are present in input files
            # only check first 10 entries as this should be enough
            for modality in test_dataset_metadata['modalities']:
                for i in range(10):
                    assert modality in split_data['input_files'][i]


def test_percent_split_generation(test_dataset, percent_split_generator, percent_split, test_dataset_metadata):

    percent_split_generator.generate_splits()

    assert (test_dataset / 'test.json').exists()
    assert (test_dataset / 'train.json').exists()

    for split, split_size in percent_split.items():
        with (test_dataset / f'{split}.json').open('r') as f:
            split_data = json.load(f)

            assert len(split_data['input_files']) == int(split_size * test_dataset_metadata['num_total_samples'])
            assert len(split_data['label_files']) == int(split_size * test_dataset_metadata['num_total_samples'])

            # check if modalities are present in input files
            # only check first 10 entries as this should be enough
            for modality in test_dataset_metadata['modalities']:
                for i in range(10):
                    assert modality in split_data['input_files'][i]
