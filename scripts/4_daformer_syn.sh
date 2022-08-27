CUDA_VISIBLE_DEVICES=4 python -m tools.train configs/daformer/syn_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/syn_to_zerov2_daformer_mit5/
CUDA_VISIBLE_DEVICES=4 python -m tools.test configs/daformer/syn_to_zerov2_daformer_mit5.py experiment/baseline/syn_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/syn_to_zerov2_daformer_mit5/predictions --opacity 1
CUDA_VISIBLE_DEVICES=4 python -m tools.convert_visuals_to_labels experiment/baseline/syn_to_zerov2_daformer_mit5/predictions experiment/baseline/syn_to_zerov2_daformer_mit5/original/


CUDA_VISIBLE_DEVICES=4 python -m tools.train configs/daformer/norcs/syn_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/norcs/syn_to_zerov2_daformer_mit5/
CUDA_VISIBLE_DEVICES=4 python -m tools.test configs/daformer/norcs/syn_to_zerov2_daformer_mit5.py experiment/baseline/norcs/syn_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/norcs/syn_to_zerov2_daformer_mit5/predictions --opacity 1
CUDA_VISIBLE_DEVICES=4 python -m tools.convert_visuals_to_labels experiment/baseline/norcs/syn_to_zerov2_daformer_mit5/predictions experiment/baseline/norcs/syn_to_zerov2_daformer_mit5/original/
