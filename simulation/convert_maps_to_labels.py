import numpy as np
import imageio 
import os, glob
import shutil
import PIL
import argparse
import tqdm

opj = os.path.join

CLASS_DICT = {
    51.: 1,
    102.: 2, 
    153.: 3,
    204.: 4,
    255.: 5}


def cleanup_semseg(sem_seg):
  oned_sem_seg = np.mean(sem_seg[..., :-1], axis=-1)
  clean_sem_seg = np.zeros_like(oned_sem_seg)
  for val in CLASS_DICT:
    class_map = oned_sem_seg == val
    clean_sem_seg[class_map] = CLASS_DICT[val]
  return clean_sem_seg

def cleanup_instseg(inst_seg):
  def add_to_channel(channel_2d, shift=255):
    nonzero_mask = channel_2d > 0
    new_channel = channel_2d + shift * nonzero_mask
    return new_channel
    
  oned_inst_seg = inst_seg[..., 0] + add_to_channel(inst_seg[..., 1], 255) \
                  + add_to_channel(inst_seg[..., 2], 511)
  return oned_inst_seg


def get_index_by_img_name(fname):
  bname = os.path.basename(fname)
  img_id = bname.split("vanilla_")[1]
  return img_id


def get_segmentations(img_fname, data_root):
  img_id = get_index_by_img_name(img_fname)
  inst_seg_name = opj(data_root, "instancecoco_%s" % img_id)
  sem_seg_name = opj(data_root, "semanticcoco_%s" % img_id)
  img = imageio.imread(img_fname)
  sem_seg = imageio.imread(sem_seg_name)
  inst_seg = imageio.imread(inst_seg_name)
  clean_instmask = cleanup_instseg(inst_seg)
  clean_segmask = cleanup_semseg(sem_seg)
  return img, img_id, clean_segmask, clean_instmask


def save_clean_segmentations(img_list, data_root, out_root):
  os.makedirs(opj(out_root, "data"), exist_ok=True)
  os.makedirs(opj(out_root, "sem_seg"), exist_ok=True)
  os.makedirs(opj(out_root, "inst_seg"), exist_ok=True)
  for img_name in tqdm.tqdm(img_list):
    try:
      img, img_id, segmask, instmask = get_segmentations(img_name, data_root)
      imageio.imsave(opj(out_root, "data", img_id), img)
      imageio.imsave(opj(out_root, "sem_seg", img_id),
                     segmask.astype(np.uint8))
      imageio.imsave(opj(out_root, "inst_seg", img_id),
                     instmask.astype(np.uint8))
    except FileNotFoundError:
      print("Cannot find the file %s" % img_name)



def main():
    parser = argparse.ArgumentParser(description='Convert simulation results to clean labels.')
    parser.add_argument('sim_folder', type=str, 
                        help='path to the folder with the simulated data.')
    parser.add_argument('out_folder', type=str, 
                        help='output dataset path')
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)
    img_list = glob.glob(opj(args.sim_folder, "vanilla_*.png"))
    print("Found %i simulation frames. Processing..." % len(img_list))
    save_clean_segmentations(img_list, args.sim_folder, args.out_folder)



if __name__ == "__main__":
    main()