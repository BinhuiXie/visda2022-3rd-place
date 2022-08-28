CUDA_VISIBLE_DEVICES=1 python -m tools.train configs/source_only/zerov1_to_zerov2_segformer.json --work-dir experiment/baseline/zerov1_to_zerov2_segformer/
CUDA_VISIBLE_DEVICES=1 python -m tools.test configs/source_only/zerov1_to_zerov2_segformer.json experiment/baseline/zerov1_to_zerov2_segformer/latest.pth --format-only --show-dir experiment/baseline/zerov1_to_zerov2_segformer/predictions --opacity 1
CUDA_VISIBLE_DEVICES=1 python -m tools.convert_visuals_to_labels experiment/baseline/zerov1_to_zerov2_segformer/predictions experiment/baseline/zerov1_to_zerov2_segformer/original/



CUDA_VISIBLE_DEVICES=1 python -m tools.train configs/source_only/zerov1all_to_zerov2_segformer.json --work-dir experiment/baseline/zerov1all_to_zerov2_segformer/
CUDA_VISIBLE_DEVICES=1 python -m tools.test configs/source_only/zerov1all_to_zerov2_segformer.json experiment/baseline/zerov1all_to_zerov2_segformer/latest.pth --format-only --show-dir experiment/baseline/zerov1all_to_zerov2_segformer/predictions --opacity 1
CUDA_VISIBLE_DEVICES=1 python -m tools.convert_visuals_to_labels experiment/baseline/zerov1all_to_zerov2_segformer/predictions experiment/baseline/zerov1all_to_zerov2_segformer/original/