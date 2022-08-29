#CUDA_VISIBLE_DEVICES=7 python -m tools.train configs/daformer/synaugzerov1_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/
#CUDA_VISIBLE_DEVICES=7 python -m tools.test configs/daformer/synaugzerov1_to_zerov2_daformer_mit5.py experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/predictions --opacity 1
#CUDA_VISIBLE_DEVICES=7 python -m tools.convert_visuals_to_labels experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/synaugzerov1_to_zerov2_daformer_mit5/original/



CUDA_VISIBLE_DEVICES=7 python -m tools.train configs/daformer/norcs/synaugzerov1_to_zerov2_daformer_mit5.py --work-dir experiment/baseline/norcs/Synaugzerov1_to_zerov2_daformer_mit5/
CUDA_VISIBLE_DEVICES=7 python -m tools.test configs/daformer/norcs/synaugzerov1_to_zerov2_daformer_mit5.py experiment/baseline/norcs/Synaugzerov1_to_zerov2_daformer_mit5/latest.pth --format-only --show-dir experiment/baseline/norcs/Synaugzerov1_to_zerov2_daformer_mit5/predictions --opacity 1
CUDA_VISIBLE_DEVICES=7 python -m tools.convert_visuals_to_labels experiment/baseline/norcs/Synaugzerov1_to_zerov2_daformer_mit5/predictions experiment/baseline/norcs/Synaugzerov1_to_zerov2_daformer_mit5/original/
