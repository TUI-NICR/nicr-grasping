"""Script for generating image showcasing the different ways to convert rectgrasp to label.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from nicr_grasping.datatypes.grasp import RectangleGraspDrawingMode, RectangleGrasp

CMAP = 'Reds'
IMG_SIZE = 200


def main():

    grasp = RectangleGrasp(1, np.ones((1, 2)) * IMG_SIZE / 2, IMG_SIZE / 1.5, np.pi/5)#, IMG_SIZE / 8)

    rect_img = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    rect_img = grasp.plot(rect_img)

    num_modes = int(np.ceil(len(RectangleGraspDrawingMode) / 3))

    f, axes = plt.subplots(3, num_modes, figsize=(num_modes * 2, 8), dpi=200)

    mode_to_index = {
        'INNER_RECTANGLE': 0,
        'INNER_TENTH_RECTANGLE': 1,
        'GAUSS': 2,
        'CENTER_POINT': 3
    }

    for mi, mode in enumerate(RectangleGraspDrawingMode):

        if 'FULL_MARGIN' in mode.name:
            row_index = 2
        elif 'MARGIN' in mode.name:
            row_index = 1
        else:
            row_index = 0

        col_index = None
        for m, i in mode_to_index.items():
            if m in mode.name:
                col_index = i
                break

        if col_index is None:
            continue

        img = np.zeros((IMG_SIZE, IMG_SIZE))
        labels = [img.copy(), img.copy(), img.copy()]
        labels = grasp.draw_label(labels, mode=mode)

        if -1 in np.unique(labels[0]):
            # scale image to [0 ... 255]
            # with negative labels the original interval is [-1 ... 1]
            quality_img = (labels[0] + 1) # *  255 / 2
        else:
            quality_img = labels[0] # *  255
        # quality_img = quality_img.astype(np.uint8)
        # quality_img = np.expand_dims(quality_img, -1)
        # quality_img = np.concatenate([quality_img, quality_img, quality_img], axis=-1)

        grasp_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        example_img = grasp.plot(grasp_img)
        alphas = example_img.sum(axis=-1) != 0
        alphas = alphas.astype(int) * 255
        alphas = np.expand_dims(alphas, -1)

        example_img = np.concatenate([example_img, alphas], axis=-1)

        axes[row_index, col_index].imshow(quality_img, cmap=plt.get_cmap(CMAP))
        axes[row_index, col_index].imshow(example_img)
        axes[row_index, col_index].set_title(mode.name, fontsize=6)

    ax_flat = [a for ax in axes for a in ax]
    for ax in ax_flat:
        ax.axis('off')

    legend_data = [
        [np.array(plt.get_cmap(CMAP)(0))[:3] * 255, 'Bad grasp'],
        [np.array(plt.get_cmap(CMAP)(255))[:3] * 255, 'Good grasp'],
        [np.array(plt.get_cmap(CMAP)(125))[:3] * 255, 'Void']
    ]

    handles = [
        Rectangle((0,0),1,1, color = tuple(v/255 for v in c)) for c,n in legend_data
    ]
    labels = list([l for _, l in legend_data])

    legend = f.legend(handles, labels, loc='lower center', ncol=3)
    legend.get_frame().set_facecolor('lightgrey')

    f.tight_layout()

    f.savefig('grasp_label_generation_example.png')
    # plt.show()


if __name__ == '__main__':
    main()
