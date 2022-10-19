# On the Importance of Label Encoding and Uncertainty Estimation for Robotic Grasp Detection

---

__NOTICE__: graspnetAPI has a dependency on Open3D which is only available through pip for Python <=3.9.
This package can be used for newer versions of Python.

---

This repository defines the structure of the datasets used for training the models.
In addition it provides scripts for cleaning the Graspnet-1Billion datasets rectangle grasp lables as described in our paper.

## Preparing Graspnet-1Billion dataset

---

For preparing the dataset a modified version of the original code is necessary.
Changes include a fix for 2D->3D projection and additions for evaluating all grasps and saving the results.
You can find the modified code in our [fork](https://github.com/best3125/graspnetAPI.git).

---

In order to reproduce our results based on our provided code the Graspnet-1Billion dataset needs to be cleaned and converted into the correct dataset structure.
To do this you need to follow these steps (You can find `exam_eval.py` in our [fork of the original Graspnet-1Billion repository](https://github.com/best3125/graspnetAPI.git):
1. Computation of evaluation files. There are two ways of generating the files required for filtering the original grasp labels.
Either you compute them yourself, if you have the time and ressources, or you download our precomputed evaluation files.
The precomputed files are based on an earlier version of the grasp labels (before the updated version of 21.03.22) and will be updated.
    1. Computation of evaluation files.
    This includes converting the 2D rectangle grasp labels to 3D and evaluating all of them.
        1. `python scripts/run_gt_predictor /tmp/graspnet-gt-pred/` for generating 3d grasps based on rect grasps (This is important to get representative grasps for evaluation as these include projection errors)
        2. `python scripts/graspnet/run_evaluation.py /tmp/graspnet-gt-pred/ --eval-all --log-dir /tmp/graspnet-gt-eval` for evaluating all grasps. THIS NEEDS A LOT OF TIME AND RESOURCES!
    2. Download precomputed files [here](https://drive.google.com/uc?id=14MAWibyRDoT_tm5z2DP0TslsZW6qkBit) (GoogleDrive). Extract the contents to `/tmp/graspnet-gt-eval`.
3. `python scripts/graspnet/filter_labels.py /tmp/graspnet-filtered --src-dir /tmp/graspnet-gt-eval/` for selecting grasps not in collision after projection creating new and cleaned dataset
4. `python scripts/convert_dataset.py graspnet <GRASPNET_PATH> /tmp/converted_graspnet --draw-mode <DRAW_MODE> --rect-label-path /tmp/graspnet-filtered --num-worker <NUM_WORKER>` to convert the dataset into the correct format.
This will copy input images and convert grasp labels into the label images needed for training.

## Encodings

The naming scheme for the encodings (called drawing mode in the codebase) differes from the one used in the paper.

| Shape | Margin | Drawing mode |
|:---| :---| ---: |
| Gauss | None | GAUSS |
| Inner Third | None | INNER_THIRD_RECTANGLE |
| Inner Tenth | None | INNER_TENTH_RECTANGLE |
| Gauss | Yes | GAUSS_WITH_FULL_MARGIN |
| Inner Third | Yes | INNER_THIRD_RECTANGLE_FULL_WITH_MARGIN |
| Inner Tenth | Yes | INNER_TENTH_RECTANGLE_FULL_WITH_MARGIN |

Wether to use outer ignore or inner ignore is handled by a parameter when training the models, NOT during dataset generation.

## Training code and weights
Code and weights see [our detectron2 repo](https://github.com/TUI-NICR/nicr-detectron2).
