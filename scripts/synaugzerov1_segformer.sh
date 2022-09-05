## source domain: synthwaste_aug_zerowastev1
## target domain: zerowaste-v2-splits/train
# DAFormer - uda
CUDA_VISIBLE_DEVICES=5 python -m tools.train configs/daformer/synaugzerov1_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/5_class/synaugzerov1_to_zerov2_daformer_mit5/
CUDA_VISIBLE_DEVICES=5 python -m tools.test configs/daformer/synaugzerov1_to_zerov2_daformer_mit5.py experiment/baseline/5_class/synaugzerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/5_class/synaugzerov1_to_zerov2_daformer_mit5/predictions --opacity 1
CUDA_VISIBLE_DEVICES=5 python -m tools.convert_visuals_to_labels experiment/baseline/5_class/synaugzerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/5_class/synaugzerov1_to_zerov2_daformer_mit5/original/

