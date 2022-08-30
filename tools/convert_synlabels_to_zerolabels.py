"""
Create by Binhui Xie (binhuixie@bit.edu.cn)
`synthwaste` -> sem_seg: 0, 1, 2, 3, 4, 5
`zerowaste` -> sem_seg: 0, 1, 2, 3, 4
Here, we change the annotation of `synthwaste` 5 to 0 as `zerowaste`
"""

import os
import numpy as np
import argparse
import imageio
import tqdm

# CLASSES = ('background', 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic', 'paper')
PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [255, 255, 0]]


def convert_seg(label_img):
    """Converts single-channel label images to RGB visual examples."""
    vis_img = np.expand_dims(np.zeros(label_img.shape), -1)
    vis_img = np.repeat(vis_img, 3, axis=-1)
    for lbl, color in enumerate(PALETTE):
        vis_mask = label_img == lbl
        vis_img[vis_mask] = color
    return vis_img


def convert_label(label_img):
    """Converts class 5 to class 0."""
    label_img[label_img == 5] = 0
    return label_img


def main():
    parser = argparse.ArgumentParser(description='Convert ZeroWaste visuals to labels.')
    parser.add_argument('label_folder', type=str,
                        default='data/synthwaste/train/sem_seg',
                        help='path to the folder with ground-truth labels.')
    parser.add_argument('out_folder', type=str,
                        default='data/synthwaste/train/sem_seg_zero',
                        help='output path with visual labels.')
    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    img_list = os.listdir(args.label_folder)

    for img_name in tqdm.tqdm(img_list):
        label_img = imageio.imread(os.path.join(args.label_folder, img_name))
        convert_lbl_img = convert_label(label_img)
        imageio.imsave(os.path.join(args.out_folder, img_name), convert_lbl_img.astype(np.uint8))


if __name__ == "__main__":
    main()


# python tools/convert_synlabels_to_zerolabels.py data/synthwaste/train/sem_seg data/synthwaste/train/sem_seg_zero
# python tools/convert_synlabels_to_zerolabels.py data/synthwaste/val/sem_seg data/synthwaste/val/sem_seg_zero
# python tools/convert_synlabels_to_zerolabels.py data/synthwaste/test/sem_seg data/synthwaste/test/sem_seg_zero
# python tools/convert_synlabels_to_zerolabels.py data/synthwaste-aug/sem_seg data/synthwaste-aug/sem_seg_zero
