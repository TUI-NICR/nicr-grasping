# GraspTrack: Object and Grasp Pose Tracking of Arbitrary Objects

We provide code for evaluating grasp poses for each object instead of the whole frame.
Additionally, we provide model weights for EMSANet used in our work.

The remaining weights used in this work are already published:
* GRConvNet: Provided in our nicr-detectron2 repository (see [here](https://github.com/TUI-NICR/nicr-detectron2/blob/master/projects/GraspEncodingsAndUncertainties/README.md#models-without-uncertainty-estimation))
* LFNet: Indoor model provided in the [official repository](https://github.com/vcg-uvic/lf-net-release/tree/master#pretrained-models-and-example-dataset)

## Evaluation

We reimplemented the evaluation pipeline from GraspNet-1Billion as the original code was not designed to used predifined assignments of grasps and objects.
In the original pipeline a grasp was assigned its object based on the distance to the pointcloud.
For evaluating grasp quality we use the same code as graspnetAPI based on dexnet therefore computed scores remain the same.

The `eval_scene.py` script can be used to evaluate grasps saved in the graspnetAPI format while taking into account the predifined object assignments.

In addition our evaluation script saves results in form of CSV files, which can be used for better understand details of the evaluation (e.g. how many grasps are present per object, which grasps where suppressed by the NMS).

## Model Weights

We provide the network weights for EMSANet trained on GraspNet-1Billion as described in our paper.
Weights can be found [here](https://drive.google.com/uc?id=1QqJLo7QnLzKz-xfJH0pyLixZ4bImeWYN).

To use these weights we provide additional code as EMSANet needs some information (e.g. depth statistics) for preprocessing.
To use the [provided files](emsanet) simply clone the [official EMSANet repository](https://github.com/TUI-NICR/EMSANet) and follow the installation guide.
The scripts can then be copied to the EMSANet code and executed.

The `graspnet_dataset.py` contains the dataset class.
This class is only for providing information such as depth statistics.

`inference_samples_graspnet.py` can be used to compute outputs of the trained EMSANet.
The script assumes a folder for depth and rgb data where files can be matched based on the filename.
E.g. to run the model over a scene of the GraspNet-1Billion dataset you can use:
```bash
WEIGHTS_FILEPATH="<PATH_TO_DOWNLOADED_WEIGHTS>"

DATA_PATH="<GRASPNET_ROOT_PATH>/scenes/scene_0100/kinect/"

OUTPUT_PATH="<PATH_WHERE_OUTPUTS_ARE_TO_BE_SAVED>"

python inference_samples_tape.py \
    --dataset graspnet \
    --raw-depth \
    --tasks semantic instance \
    --enable-panoptic \
    --no-pretrained-backbone \
    --weights-filepath ${WEIGHTS_FILEPATH} \
    --context-module appm-1-2-4-8 \
    --input-height 768 \
    --input-width 1280 \
    --inference-input-height 768 \
    --inference-input-width 1280 \
    --inference-data-basepath ${DATA_PATH} \
    --instance-center-heatmap-threshold 0.1 \
    --instance-center-heatmap-nms-kernel-size 51 \
    --instance-center-heatmap-top-k 64 \
    --instance-offset-distance-threshold 50 \
    --output-basepath ${OUTPUT_PATH}
```

__NOTE__: The example call contains all parameters used for the instance segmentation done for evaluation on GraspNet-1Billion.