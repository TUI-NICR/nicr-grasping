from .dataset_info import DatasetInfo

# Split definitions
GRASPNET_SPLITS = {
    'split_sizes': {'train': 25600, 'test_seen': 7680, 'test_similar': 7680, 'test_novel': 7680},
    'composite_splits': {'test': ['test_seen', 'test_similar', 'test_novel']}
}
CORNELL_SPLITS = {
    'split_sizes': {'train': 704, 'test': 181},
    'composite_splits': {}
}

SPLIT_DEFINITIONS = {
    'graspnet': GRASPNET_SPLITS,
    'cornell': CORNELL_SPLITS
}
