## source domain: zerowaste-f/train
## target domain: zerowaste-v2-splits/train
# DAFormer - uda
### Training
CUDA_VISIBLE_DEVICES=3 python -m tools.train configs/daformer/zerov1_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/zerov1_to_zerov2_daformer_mit5/

### Testing
CUDA_VISIBLE_DEVICES=3 python -m tools.test configs/daformer/zerov1_to_zerov2_daformer_mit5.py experiment/baseline/zerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/zerov1_to_zerov2_daformer_mit5/predictions --opacity 1

### Predictions
CUDA_VISIBLE_DEVICES=3 python -m tools.convert_visuals_to_labels experiment/baseline/zerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/zerov1_to_zerov2_daformer_mit5/original/

