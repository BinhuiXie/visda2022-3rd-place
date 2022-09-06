## source domain: synthwaste_aug_zerowastev1
## target domain: zerowaste-v2-splits/train
# SegFormer - source only
#### Training
#CUDA_VISIBLE_DEVICES=5 python -m tools.train configs/source_only/synaugzerov1_to_zerov2_segformer.py --work-dir experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_segformer/
#
#### Testing
#CUDA_VISIBLE_DEVICES=5 python -m tools.test configs/source_only/synaugzerov1_to_zerov2_segformer.py experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_segformer/latest.pth --format-only --show-dir experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_segformer/predictions --opacity 1
#
#### Predictions
#CUDA_VISIBLE_DEVICES=5 python -m tools.convert_visuals_to_labels experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_segformer/predictions experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_segformer/original/



# DAFormer - uda
### Training
CUDA_VISIBLE_DEVICES=5 python -m tools.train configs/daformer/5_class_noFD/synaugzerov1_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_daformer_mit5/

### Testing
CUDA_VISIBLE_DEVICES=5 python -m tools.test configs/daformer/5_class_noFD/synaugzerov1_to_zerov2_daformer_mit5.py experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_daformer_mit5/predictions --opacity 1

### Predictions
CUDA_VISIBLE_DEVICES=5 python -m tools.convert_visuals_to_labels experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/5_class_noFD/synaugzerov1_to_zerov2_daformer_mit5/original/

