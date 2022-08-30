import os
import numpy as np
import argparse
import imageio
import tqdm

# ZeroWaste palette
# PALETTE = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156]]
PALETTE = [[0, 0, 0], [0, 0, 230], [250, 170, 30], [153, 153, 153], [220, 220, 0]]

def convert_seg(vis_img):
    label_img = np.zeros([*vis_img.shape[:-1]])
    for idx, lbl in enumerate(PALETTE):
        lbl_mask = vis_img == lbl
        label_img[lbl_mask[..., 0]] = idx
    return label_img


def main():
    """Converts RGB visual examples to single-channel label images."""
    parser = argparse.ArgumentParser(description='Convert ZeroWaste visuals to labels.')
    parser.add_argument('vis_folder', type=str,
                        help='path to the folder with predicted visuals.')
    parser.add_argument('out_folder', type=str,
                        help='output path with predicted labels.')
    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)
    img_list = os.listdir(args.vis_folder)
    for img_name in tqdm.tqdm(img_list):
        pred_img = imageio.imread(os.path.join(args.vis_folder, img_name))
        pred_lbl_img = convert_seg(pred_img)
        imageio.imsave(
            os.path.join(args.out_folder, img_name),
            pred_lbl_img.astype(np.uint8))


if __name__ == "__main__":
    main()
