## source domain: zerowaste-f/train
## target domain: zerowaste-v2-splits/train
# fst_d - uda
### Training
CUDA_VISIBLE_DEVICES=4 python -m tools.train configs/fst_d/zerowastev12v2_uda_warm_fdthings_rcs_croppl_a999_nesterov_mitb5_s0_step3.py --work-dir experiment/zerov1_to_zerov2_fst_d_mit5/

### Testing
CUDA_VISIBLE_DEVICES=4 python -m tools.test configs/fst_d/zerowastev12v2_uda_warm_fdthings_rcs_croppl_a999_nesterov_mitb5_s0_step3.py experiment/zerov1_to_zerov2_fst_d_mit5/latest.pth --format-only --show-dir experiment/zerov1_to_zerov2_fst_d_mit5/predictions --opacity 1

### Predictions
CUDA_VISIBLE_DEVICES=4 python -m tools.convert_visuals_to_labels experiment/zerov1_to_zerov2_fst_d_mit5/predictions experiment/zerov1_to_zerov2_fst_d_mit5/original/

