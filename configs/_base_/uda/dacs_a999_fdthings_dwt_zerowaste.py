# UDA with Thing-Class ImageNet Feature Distance + Increased Alpha
_base_ = ['dacs.py']
uda = dict(
    type='DACSDWTMix',
    alpha=0.999,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[1, 2, 3, 4],
    imnet_feature_dist_scale_min_ratio=0.75,
    dwt_alpha=0.5,
)
