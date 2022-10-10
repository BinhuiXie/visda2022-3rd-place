# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, single_gpu_test
from .train import (get_root_logger, set_random_seed,
                    train_segmentor)
from .ensemble_test import multi_gpu_test_ensemble, single_gpu_test_ensemble

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test',
    'show_result_pyplot', 'multi_gpu_test_ensemble',
    'single_gpu_test_ensemble'
]
