# Dataset format
This file describes the format defined for datasets in this pakage.

## Folder structure
This pakage defines the folder structure of its generated datasets as follows:
```
<dataset_name>
│   metadata.json
│
└── grasp_labels
│   │   *_angle.npz
│   │   *_quality.npz
│   │   *_width.npz
│
└── grasp_lists
│   │   *.npy
│
└── <modality>
│   │   *.<modality_fileending>
```

Where `<dataset_name>` is the name specified by the interface (e.g. 'cornell' or 'graspnet'), `<modality>` represents all modalities of the dataset (e.g. if the dataset has color and depth images there will be to folders 'color_image' and 'depth_image'), `<modality_fileending>` is the respective fileending for a modality (e.g. 'png', 'tiff') and `*` stands for the sample id padded with zeros (e.g. '0001').

If splits where defined the respective files will be located in the same directory as the `metadata.json`.

### Metadata
The `metadata.json` contains the following entries:
* `num_total_samples`: Number of samples contained in the dataset (over all splits). This is used for padding the sample id.
* `conversion_data`: Timestamp when the dataset was created
* `modalities`: List of modalities. Entries can be used by `nicr_grasping.datatypes.modalities.get_modality_from_name(<modality_name>)` to get a modality object.

2D grasp labels are saved as sparse matrices and can be loaded from file with `scipy.sparse.load_npz` and converted to an array with the `toarray()` method.

### Split files
A file specifying a split contains the following entries:
* `input_files`: List of dictionaries which contain an entry for every modality. For example an element of this list could look like this:

    `{"color_image": "color_image/25600.png", "depth_image": "depth_image/25600.tiff"}`

    The files are specified with relative paths.
* `label_files`: List of dictionaries. For 2D grasps the keys are `quality`, `width` and `angle` with the relative path to the label file.

## 2D label generation
For 2D grasps there are multiple ways to convert them into labels which can be used by models such as GGCNN.
The following image shows all implemented ways of converting a grasp.
The INNER_RECTANGLE method is widly used in the literature wheras INNER_TENTH_RECTANGLE was used by GraspNet.

![](grasp_label_generation_example.png)