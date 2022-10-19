from pathlib import Path
import json
import numpy as np
import os

from typing import Dict, List, Union

from numpy.random import sample

from ..datatypes.modalities import get_modality_from_name

class SplitGenerator:
    def __init__(self, path: Union[str, Path],
                 randomize: bool = False,
                 split_percs: Dict[str,float] = None,
                 split_sizes: Dict[str,int] = None,
                 composite_splits: Dict[str, List[str]] = None) -> None:
        self._path = Path(path)

        assert split_percs is not None or split_sizes is not None
        assert self._path.exists()

        self._randomize = randomize
        self._metadata = None
        self._read_metadata()

        if split_sizes is not None and split_percs is not None:
            raise ValueError("You can only specify splits in one way. Percentages and discrete sizes were given")
        elif split_sizes is not None:
            self._split_sizes = split_sizes
        elif split_percs is not None:
            self._split_sizes = {}
            for split, perc in split_percs.items():
                self._split_sizes[split] = int(perc * self._metadata['num_total_samples'])

        self._composite_splits = composite_splits


    def _read_metadata(self) -> None:
        with (self._path / 'metadata.json').open('r') as f:
            metadata = json.load(f)
            self._metadata = metadata

    def generate_splits(self) -> None:
        sample_ids = list(range(self._metadata['num_total_samples']))

        if self._randomize:
            np.random.shuffle(sample_ids)

        modalities = self._metadata['modalities']
        len_sample_id = len(str(self._metadata['num_total_samples']))

        splits = {}

        last_used_sample_id = 0
        for split, split_size in self._split_sizes.items():
            split_data = {
                'input_files': [],
                'label_files': []
            }
            split_sample_ids = sample_ids[last_used_sample_id : (last_used_sample_id + split_size)]
            last_used_sample_id += split_size
            for s_id in split_sample_ids:
                s_id_str = str(s_id).rjust(len_sample_id, '0')

                input_files = {}

                for modality_str in modalities:
                    modality = get_modality_from_name(modality_str)
                    input_files[modality_str] = os.path.join(modality_str, s_id_str + modality.FILE_ENDING)

                split_data['input_files'].append(input_files)

                # TODO: make it more general for 3d labels

                output_files = {}
                for label_type in ['quality', 'angle', 'width']:
                    output_files[label_type] = os.path.join('grasp_labels', s_id_str + '_' + label_type + '.npz')

                split_data['label_files'].append(output_files)

            splits[split] = split_data

        if self._composite_splits is not None:
            for comp_split, parts in self._composite_splits.items():
                split_data = {
                    'input_files': [],
                    'label_files': []
                }

                for part in parts:
                    split_data['input_files'] += splits[part]['input_files']
                    split_data['label_files'] += splits[part]['label_files']

                splits[comp_split] = split_data

        for split, split_data in splits.items():
            with (self._path / (split + '.json')).open('w') as f:
                json.dump(split_data, f)





