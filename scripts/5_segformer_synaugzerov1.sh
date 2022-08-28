CUDA_VISIBLE_DEVICES=5 python -m tools.train configs/source_only/synaugzerov1_to_zerov2_segformer.json --work-dir experiment/baseline/synaugzerov1_to_zerov2_segformer/
CUDA_VISIBLE_DEVICES=5 python -m tools.test configs/source_only/synaugzerov1_to_zerov2_segformer.json experiment/baseline/synaugzerov1_to_zerov2_segformer/latest.pth --format-only --show-dir experiment/baseline/synaugzerov1_to_zerov2_segformer/predictions --opacity 1
CUDA_VISIBLE_DEVICES=5 python -m tools.convert_visuals_to_labels experiment/baseline/synaugzerov1_to_zerov2_segformer/predictions experiment/baseline/synaugzerov1_to_zerov2_segformer/original/



CUDA_VISIBLE_DEVICES=5 python -m tools.train configs/source_only/syn_to_zerov2_segformer.json --work-dir experiment/baseline/syn_to_zerov2_segformer/
CUDA_VISIBLE_DEVICES=5 python -m tools.test configs/source_only/syn_to_zerov2_segformer.json experiment/baseline/syn_to_zerov2_segformer/latest.pth --format-only --show-dir experiment/baseline/syn_to_zerov2_segformer/predictions --opacity 1
CUDA_VISIBLE_DEVICES=5 python -m tools.convert_visuals_to_labels experiment/baseline/syn_to_zerov2_segformer/predictions experiment/baseline/syn_to_zerov2_segformer/original/