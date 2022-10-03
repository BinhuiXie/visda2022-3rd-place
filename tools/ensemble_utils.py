import numpy as np
import torch


class EnsemblePolicy:
    @classmethod
    def list_all_p(cls):
        return [x for x in dir(cls) if x.endswith('policy')]

    @classmethod
    def get_p_by_name(cls, p_name: str):
        return getattr(cls, p_name) if p_name is not None else None

    @staticmethod
    def average_policy(result_raw):  # result_raw in form list(tensor(CxHxW))
        seg_pred = torch.stack(result_raw, 0).mean(0).argmax(0, keepdim=True)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
