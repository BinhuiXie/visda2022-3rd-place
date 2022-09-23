from pytorch_wavelets import DWT, IDWT
import torch
import torch.nn.functional as F

xfm = DWT(J=1, mode='zero', wave='haar').cuda()  # Accepts all wave types available to PyWavelets
ifm = IDWT(mode='zero', wave='haar').cuda()


def dwt_copy_paste(mask, copy_img, paste_img, alpha=0.5):
    with torch.no_grad():
        Yl_copy, Yh_copy = xfm(copy_img.detach())
        Yl_paste, Yh_paste = xfm(paste_img.detach())

        Yl_mix = Yl_paste.clone()
        Yl_mix[mask] = Yl_copy[mask] * alpha + Yl_paste[mask] * (1 - alpha)
        Yh_mix = [torch.maximum(a, b) for (a, b) in zip(Yh_copy, Yh_paste)]

        mix_img = ifm((Yl_mix, Yh_mix)).detach()
    return mix_img
