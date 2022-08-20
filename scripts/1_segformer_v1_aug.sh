CUDA_VISIBLE_DEVICES=1 python -m tools.train configs/source_only/synthzerowaste_segformer.json --work-dir experiment/baseline/segformerV1Aug/
CUDA_VISIBLE_DEVICES=1 python -m tools.test configs/source_only/synthzerowaste_segformer.json experiment/baseline/segformerV1Aug/checkpoint.pth --eval mIoU --show-dir experiment/baseline/segformerV1Aug/predictions --opacity 1
CUDA_VISIBLE_DEVICES=1 python -m tools.convert_visuals_to_labels experiment/baseline/segformerV1Aug/predictions experiment/baseline/segformerV1Aug/original/
