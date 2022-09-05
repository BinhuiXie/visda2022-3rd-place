import os
import numpy as np
import argparse
import imageio
import tqdm

# ZeroWaste palette
# PALETTE = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]]
# PALETTE = [[0, 0, 0], [0, 0, 230], [250, 170, 30], [153, 153, 153], [220, 220, 0]]

# CLASSES = ('background', 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic', 'paper')
# PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [255, 255, 0]]
PALETTE = [[0, 0, 0], [0, 0, 230], [250, 170, 30], [153, 153, 153], [152, 251, 152]]


def convert_seg(label_img):
    """Converts single-channel label images to RGB visual examples."""
    vis_img = np.expand_dims(np.zeros(label_img.shape), -1)
    vis_img = np.repeat(vis_img, 3, axis=-1)
    for lbl, color in enumerate(PALETTE):
        vis_mask = label_img == lbl
        vis_img[vis_mask] = color
    return vis_img


def main():
    parser = argparse.ArgumentParser(description='Convert ZeroWaste visuals to labels.')
    parser.add_argument('label_folder', type=str, default='D:/Documents/dataset/visda2022/synthwaste_splits/train/sem_seg',
                        help='path to the folder with ground-truth labels.')
    parser.add_argument('out_folder', type=str,  default='D:/Documents/dataset/visda2022/synthwaste_splits/train/sem_seg_color',
                        help='output path with visual labels.')
    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    img_list = os.listdir(args.label_folder)

    background_list = []
    object_ratio_dict = {i: [] for i in range(11)}

    for img_name in tqdm.tqdm(img_list):
        label_img = imageio.imread(os.path.join(args.label_folder, img_name))
        pred_lbl_img = convert_seg(label_img)
        imageio.imsave(
            os.path.join(args.out_folder, img_name.replace('.PNG', '_color.PNG')),
            pred_lbl_img.astype(np.uint8))

        if label_img.max() == 0:
            background_list.append(img_name)

        object_ratio_dict[int((label_img != 0).sum() / label_img.size * 10)].append(img_name)

    print(len(background_list))
    for key, value in object_ratio_dict.items():
        print(key, len(value))

    import torch
    torch.save({'all_background': background_list, 'object_ratio_dict': object_ratio_dict},
               os.path.join(args.out_folder, 'label_statics.pth'))


if __name__ == "__main__":
    main()
