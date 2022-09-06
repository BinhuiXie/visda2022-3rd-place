## source domain: zerowaste-f/train
## target domain: zerowaste-v2-splits/train
# SegFormer - source only
#### Training
#CUDA_VISIBLE_DEVICES=7 python -m tools.train configs/source_only/zerov1_to_zerov2_segformer.json --work-dir experiment/baseline/zerov1_to_zerov2_segformer/
#
#### Testing
#CUDA_VISIBLE_DEVICES=7 python -m tools.test configs/source_only/zerov1_to_zerov2_segformer.json experiment/baseline/zerov1_to_zerov2_segformer/latest.pth --format-only --show-dir experiment/baseline/zerov1_to_zerov2_segformer/predictions --opacity 1
#
#### Predictions
#CUDA_VISIBLE_DEVICES=7 python -m tools.convert_visuals_to_labels experiment/baseline/zerov1_to_zerov2_segformer/predictions experiment/baseline/zerov1_to_zerov2_segformer/original/



# DAFormer - uda
### Training
CUDA_VISIBLE_DEVICES=7 python -m tools.train configs/daformer/5_class_noFD/zerov1_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/5_class_noFD/zerov1_to_zerov2_daformer_mit5/

### Testing
CUDA_VISIBLE_DEVICES=7 python -m tools.test configs/daformer/5_class_noFD/zerov1_to_zerov2_daformer_mit5.py experiment/baseline/5_class_noFD/zerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/5_class_noFD/zerov1_to_zerov2_daformer_mit5/predictions --opacity 1

### Predictions
CUDA_VISIBLE_DEVICES=7 python -m tools.convert_visuals_to_labels experiment/baseline/5_class_noFD/zerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/5_class_noFD/zerov1_to_zerov2_daformer_mit5/original/

