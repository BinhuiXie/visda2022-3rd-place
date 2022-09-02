## source domain: synthwaste_aug_zerowastev1
## target domain: zerowaste-v2-splits/train
# SegFormer - source only
#CUDA_VISIBLE_DEVICES=7 python -m tools.train configs/source_only/synaugzerov1_to_zerov2_segformer_convert-clean.json --work-dir experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_segformer/
#CUDA_VISIBLE_DEVICES=7 python -m tools.test configs/source_only/synaugzerov1_to_zerov2_segformer_convert-clean.json experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_segformer/latest.pth --format-only --show-dir experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_segformer/predictions --opacity 1
#CUDA_VISIBLE_DEVICES=7 python -m tools.convert_visuals_to_labels experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_segformer/predictions experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_segformer/original/

# DAFormer - uda
CUDA_VISIBLE_DEVICES=7 python -m tools.train configs/daformer/synaugzerov1_to_zerov2_daformer_mit5_convert-clean.py --work-dir experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_daformer_mit5/
CUDA_VISIBLE_DEVICES=7 python -m tools.test configs/daformer/synaugzerov1_to_zerov2_daformer_mit5_convert-clean.py experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_daformer_mit5/predictions --opacity 1
CUDA_VISIBLE_DEVICES=7 python -m tools.convert_visuals_to_labels experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/5_class_convert-clean/synaugzerov1_to_zerov2_daformer_mit5/original/

