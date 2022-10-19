# Evaluation
## GraspNet
This section describes how a model can be evaluated through the [GraspNet](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf) pipeline.

The general process consists of the following steps:

1. Train your model on GraspNet dataset.

2. Save predictions of model in correct folder structure.

3. Run evaluation.

### Training your model
This step can be implemented any way.

---
**NOTE**

Current scripts expect a model trained with detectron2.

---

### Save predicions.
The folder structure has to follow the GraspNet format (scene_<scene_id>/\<camera>/\<annotation_id>.npy) and need to contain grasps in GraspNet format.

For models trained with detectron2 you can use `run_on_graspnet.py` as follows:
```bash
python run_on_graspnet.py --config <PATH_TO_DETECTRON2_CONFIG>
```
This will load the model and weights called `model_final.pth` in the same folder as the config file and compute predictions on the test set of graspnet.
If you want to specify different weights you can do so by adding `MODEL.WEIGHTS <PATH_TO_WEIGHTS>` as command line parameters.

### Run evaluation

For evaluation of predictions the `run_evaluation.py` script can be used.
```
usage: run_evaluation.py [-h] [--eval-all] [--top-k TOP_K]
                         [--scene-id SCENE_ID] [--camera {kinect,realsense}]
                         [--split {train,test,test_seen,test_novel,test_similar}]
                         [--log-eval] [--num-worker NUM_WORKER]
                         src_folder

positional arguments:
  src_folder            Folder from which the grasps will be loaded.

optional arguments:
  -h, --help            show this help message and exit
  --eval-all
  --top-k TOP_K         Number of top grasps to use for evaluation. Ignored if
                        --eval-all is used.
  --scene-id SCENE_ID   If supplied onyl this scene will be evaluated.
  --camera {kinect,realsense}
  --split {train,test,test_seen,test_novel,test_similar}
  --log-eval            If supplied logs will be saved here as json.
  --num-worker NUM_WORKER
```

---

**NOTE**

As the evaluation is quite resource hungy it should only be used of devices with a high cpu core count and RAM.

---

**NOTE2**

When using `--log-eval` make sure the permissions of the folder with the predictions are correct.
Otherwise the logfiles will not be saved and workers might crash.

---

