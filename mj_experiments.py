import itertools
import logging
import math


def get_model_base(architecture, backbone):
    architecture = architecture.replace('sfa_', '')
    architecture = architecture.replace('_nodbn', '')
    if 'segformer' in architecture:
        return {
            'mitb5': f'_base_/models/{architecture}_b5.py',
            # It's intended that <=b4 refers to b5 config
            'mitb4': f'_base_/models/{architecture}_b5.py',
            'mitb3': f'_base_/models/{architecture}_b5.py',
            'r101v1c': f'_base_/models/{architecture}_r101.py',
        }[backbone]
    if 'daformer_' in architecture and 'mitb5' in backbone:
        return f'_base_/models/{architecture}_mitb5.py'
    if 'upernet' in architecture and 'mit' in backbone:
        return f'_base_/models/{architecture}_mit.py'
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2red': '_base_/models/deeplabv2red_r50-d8.py',
        'dlv3p': '_base_/models/deeplabv3plus_r50-d8.py',
        'da': '_base_/models/danet_r50-d8.py',
        'isa': '_base_/models/isanet_r50-d8.py',
        'uper': '_base_/models/upernet_r50.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
        'x50-32': 'open-mmlab://resnext50_32x4d',
        'x101-32': 'open-mmlab://resnext101_32x4d',
        's50': 'open-mmlab://resnest50',
        's101': 'open-mmlab://resnest101',
        's200': 'open-mmlab://resnest200',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
        'x50-32': {
            'type': 'ResNeXt',
            'depth': 50,
            'groups': 32,
            'base_width': 4,
        },
        'x101-32': {
            'type': 'ResNeXt',
            'depth': 101,
            'groups': 32,
            'base_width': 4,
        },
        's50': {
            'type': 'ResNeSt',
            'depth': 50,
            'stem_channels': 64,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's101': {
            'type': 'ResNeSt',
            'depth': 101,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True
        },
        's200': {
            'type': 'ResNeSt',
            'depth': 200,
            'stem_channels': 128,
            'radix': 2,
            'reduction_factor': 4,
            'avg_down_stride': True,
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    if 'dlv3p' in architecture and 'mit' in backbone:
        cfg['model']['decode_head']['c1_in_channels'] = 64
    if 'sfa' in architecture:
        cfg['model']['decode_head']['in_channels'] = 512
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        if 'sfa_' in architecture_mod:
            cfg['model']['neck'] = dict(type='SegFormerAdapter')
        if '_nodbn' in architecture_mod:
            cfg['model'].setdefault('decode_head', {})
            cfg['model']['decode_head']['norm_cfg'] = None
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        if uda == 'target-only':
            cfg['_base_'].append(f'_base_/datasets/{target}_half_{crop}.py')
        elif uda == 'source-only':
            cfg['_base_'].append(
                f'_base_/datasets/{source}_to_{target}_{crop}.py')
        else:
            cfg['_base_'].append(
                f'_base_/datasets/uda_{source}_to_{target}_{crop}.py')
            cfg['_base_'].append(f'_base_/uda/{uda}.py')

        cfg.setdefault('uda', {})
        if method_name in uda and plcrop:
            cfg['uda']['pseudo_weight_ignore_top'] = 15
            cfg['uda']['pseudo_weight_ignore_bottom'] = 120
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        if method_name in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)

        if method_name in uda:
            cfg['uda']['start_distribution_iter'] = start_distribution_iter
            if use_bank:
                cfg['uda']['memory_length'] = memory_length

            cfg['model'].setdefault('decode_head', {})
            cfg['model'].setdefault('auxiliary_head', {})
            cfg['model']['auxiliary_head']['in_channels'] = in_channels
            cfg['model']['auxiliary_head']['in_index'] = contrast_indexes
            cfg['model']['auxiliary_head']['input_transform'] = contrast_mode
            cfg['model']['auxiliary_head']['channels'] = channels
            cfg['model']['auxiliary_head']['num_convs'] = num_convs
            if num_convs == 0:
                if contrast_mode == 'resize_concat':
                    cfg['model']['auxiliary_head']['channels'] = sum(in_channels)
                else:
                    cfg['model']['auxiliary_head']['channels'] = in_channels
            cfg['model']['auxiliary_head'].setdefault('loss_decode', {})
            cfg['model']['auxiliary_head']['loss_decode']['use_dist'] = use_dist
            cfg['model']['auxiliary_head']['loss_decode']['use_bank'] = use_bank
            cfg['model']['auxiliary_head']['loss_decode']['use_reg'] = use_reg
            cfg['model']['auxiliary_head']['loss_decode']['use_avg_pool'] = use_avg_pool
            cfg['model']['auxiliary_head']['loss_decode']['scale_min_ratio'] = scale_min_ratio
            cfg['model']['auxiliary_head']['loss_decode']['contrast_temp'] = contrastive_temperature
            cfg['model']['auxiliary_head']['loss_decode']['loss_weight'] = contrastive_weight
            cfg['model']['auxiliary_head']['loss_decode']['reg_relative_weight'] = reg_relative_weight

        if method_name in uda and enable_self_training:
            cfg['uda']['enable_self_training'] = enable_self_training

        # Setup optimizer and schedule
        if method_name in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=iters // 10, metric='mIoU')

        # Construct config name
        uda_mod = uda

        if method_name in uda:
            if use_dist:
                uda_mod += '_DistCL'
            elif use_bank:
                uda_mod += '_BankCL'
            else:
                uda_mod += '_ProtoCL'
            if use_reg:
                uda_mod += f'-reg-w{reg_relative_weight * contrastive_weight}'
            uda_mod += f'-start-iter{start_distribution_iter}'
            uda_mod += f'-tau{contrastive_temperature}'
            if contrast_mode == 'multiple_select':
                for lyr in contrast_indexes:
                    uda_mod += f'-l{lyr}-w{contrastive_weight}'
            else:
                uda_mod += f'-l{contrast_indexes}-w{contrastive_weight}'

        if method_name in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'
        if method_name in uda and plcrop:
            uda_mod += '_cpl'
        if method_name in uda and enable_self_training:
            uda_mod += '_self'

        # Construct config name
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        cfg['name'] = f"{cfg['name_architecture']}_{cfg['name_uda']}_{cfg['name_opt']}_{cfg['name_dataset']}"
        if seed is not None:
            cfg['name'] += f'_seed{seed}'
        cfg['name'] = cfg['name'].replace('.', '.').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'cs') \
            .replace('synthia', 'syn')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    method_name = 'sepico'
    n_gpus = 1
    batch_size = 2
    iters = 40000
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '640x640'
    datasets = [
        ('zerowaste', 'zerowaste'),
    ]

    rcs_T = None
    plcrop = True
    start_distribution_iter = 3000
    enable_self_training = True

    # auxiliary head parameters
    in_channels = 2048  # in_channels = [256, 512, 1024, 2048]
    channels = 512  # default out_dim
    num_convs = 2
    contrast_indexes = 3  # int or list, depending on value of contrast_mode
    contrast_mode = None  # optional(None, 'resize_concat', 'multiple_select')
    use_dist = False
    use_bank = False
    memory_length = 200
    use_reg = False
    use_avg_pool = True
    scale_min_ratio = 0.75  # used for down-sampling
    contrastive_temperature = 100.
    contrastive_weight = 1.0
    reg_relative_weight = 1.0  # reg_weight = reg_relative_weight * loss_weight in auxiliary head

    architecture = None
    workers_per_gpu = 4
    # -------------------------------------------------------------------------
    # VisDA official baseline: seed=0 official#v2val{50.84}; seed=0 repro#v2val{51.69}
    # -------------------------------------------------------------------------
    if id == -1:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings'
        crop = '512x512'
        # rcs_T = 0.01
        plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # VisDA official baseline: seed=0 official#v2val{50.84}; seed=0 repro#v2val{51.69}
    # -------------------------------------------------------------------------
    elif id == 0:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings'
        # crop = '512x512'
        # rcs_T = 0.01
        plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # why use plcrop? !!!WRONG FD
    # -------------------------------------------------------------------------
    elif id == 1:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings'
        # rcs_T = 0.01
        # plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # try LogitConstraint loss! !!!WRONG FD
    # -------------------------------------------------------------------------
    elif id == 2:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp_lc', 'mitb5')
        uda = 'dacs_a999_fdthings'
        # rcs_T = 0.01
        # plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # will there be conflicts?
    # -------------------------------------------------------------------------
    elif id == 3:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp_lc', 'mitb5')
        uda = 'dacs_a999'
        # rcs_T = 0.01
        # plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # why use plcrop?
    # -------------------------------------------------------------------------
    elif id == 4:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings_zerowaste'
        # rcs_T = 0.01
        # plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # try LogitConstraint loss!
    # -------------------------------------------------------------------------
    elif id == 5:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp_lc', 'mitb5')
        uda = 'dacs_a999_fdthings_zerowaste'
        # rcs_T = 0.01
        # plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # try best baseline again!  dacs_a999_fdthings_zerowaste!!!!
    # -------------------------------------------------------------------------
    elif id == 6:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp', 'mitb5')
        uda = 'dacs_a999_fdthings_zerowaste'
        # rcs_T = 0.01
        plcrop = True
        for (source, target), seed in \
                itertools.product(datasets, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # try sepico (DistCL)
    # -------------------------------------------------------------------------
    elif id == 7:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp_proj', 'mitb5')
        uda = 'sepico'
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (True, False),  # DistCL
        ]
        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # try sepico (BankCL)
    # -------------------------------------------------------------------------
    elif id == 8:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp_proj', 'mitb5')
        uda = 'sepico'
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (False, True),  # BankCL
        ]
        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # try sepico (ProtoCL)
    # -------------------------------------------------------------------------
    elif id == 9:
        seeds = [0]
        datasets = [
            ('zerov1', 'zerov2'),
        ]
        architecture, backbone = ('daformer_sepaspp_proj', 'mitb5')
        uda = 'sepico'
        # aux
        num_convs = 2
        modes = [
            # in_channels, contrast_indexes, contrast_mode
            ([64, 128, 320, 512], [0, 1, 2, 3], 'resize_concat'),  # fusion
        ]
        # reg
        use_reg = True
        reg_relative_weight = 1.0
        # contrastive variants
        methods = [
            # use_dist, use_bank
            (False, False),  # ProtoCL
        ]
        # results
        for seed, mode, (use_dist, use_bank), (source, target) in itertools.product(seeds, modes, methods, datasets):
            in_channels, contrast_indexes, contrast_mode = mode
            cfg = config_from_vars()
            cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
