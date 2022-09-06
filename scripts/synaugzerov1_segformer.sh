## source domain: synthwaste_aug_zerowastev1
## target domain: zerowaste-v2-splits/train
# DAFormer - uda
### Training
CUDA_VISIBLE_DEVICES=5 python -m tools.train configs/daformer/synaugzerov1_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/

### Testing
CUDA_VISIBLE_DEVICES=5 python -m tools.test configs/daformer/synaugzerov1_to_zerov2_daformer_mit5.py experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/predictions --opacity 1

### Predictions
CUDA_VISIBLE_DEVICES=5 python -m tools.convert_visuals_to_labels experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/original/

