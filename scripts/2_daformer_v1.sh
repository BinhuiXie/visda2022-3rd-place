CUDA_VISIBLE_DEVICES=2 python -m tools.train configs/daformer/zerowaste_to_zerowastev2_daformer_mit5.py --work-dir experiment/baseline/daformerV1/
CUDA_VISIBLE_DEVICES=2 python -m tools.test configs/daformer/zerowaste_to_zerowastev2_daformer_mit5.py experiment/baseline/daformerV1/checkpoint.pth --eval mIoU --show-dir experiment/baseline/daformerV1/predictions --opacity 1
CUDA_VISIBLE_DEVICES=2 python -m tools.convert_visuals_to_labels experiment/baseline/daformerV1/predictions experiment/baseline/daformerV1/original/
