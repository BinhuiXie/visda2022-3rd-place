# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(960, 540)),  # original 1920x1080
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 540),  # original 1920x1080
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='SynthWasteDataset',
        data_root='data/synthwaste_aug_zerowastev1/',
        img_dir='clean_data',
        ann_dir='sem_seg',
        pipeline=train_pipeline),
    val=dict(
        type='ZeroWasteDataset',
        data_root='data/zerowaste-f/test',
        img_dir='data',
        ann_dir='sem_seg',
        pipeline=test_pipeline),
    # Phase 1 evaluate on zerowaste-v2 val set
    test=dict(
       type='ZeroWasteV2Dataset',
            data_root='data/zerowaste-v2-splits/val',
            img_dir='data',
            ann_dir='sem_seg',
            pipeline=test_pipeline)

    # Final evaluate on zerowaste-v2 test set
    # test = dict(
    #     type='ZeroWasteV2Dataset',
    #     data_root='data/zerowaste-v2-splits/val',
    #     img_dir='data',
    #     ann_dir='sem_seg',
    #     pipeline=test_pipeline)
)
