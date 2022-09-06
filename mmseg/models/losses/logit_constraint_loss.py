# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class LogitConstraintLoss(CrossEntropyLoss):
    """CrossEntropyLoss after Logit Norm.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 eps=1e-7):
        super(LogitConstraintLoss, self).__init__(use_sigmoid,
                                                  use_mask,
                                                  reduction,
                                                  class_weight,
                                                  loss_weight)
        self.eps = eps

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        norms = torch.norm(cls_score, p=2, dim=1, keepdim=True) + self.eps
        normed_logit = torch.div(cls_score, norms)
        loss_cls = super(LogitConstraintLoss, self).forward(normed_logit,
                                                            label,
                                                            weight,
                                                            avg_factor,
                                                            reduction_override,
                                                            **kwargs)
        return loss_cls
