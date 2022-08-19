#!/bin/bash

# Automatic Downloads:
set -e  # exit when any command fails
mkdir -p pretrained/
cd pretrained/
wget http://csr.bu.edu/ftp/recycle/visda-2022/checkpoints/mit_b5.pth # ImageNet-pretrained SegFormer
wget http://csr.bu.edu/ftp/recycle/visda-2022/checkpoints/synthwaste.pth # SynthWaste-pretrained SegFormer
wget http://csr.bu.edu/ftp/recycle/visda-2022/checkpoints/synthwaste_aug.pth # SegFormer trained on SynthWaste-aug
wget http://csr.bu.edu/ftp/recycle/visda-2022/checkpoints/uda_zerowaste2zerowastev2.pth # DAFormer trained on ZeroWaste V1 -> ZeroWaste V2
wget http://csr.bu.edu/ftp/recycle/visda-2022/checkpoints/source_only_segformer_zerowaste.pth # SegFormer trained on ZeroWaste V1
wget http://csr.bu.edu/ftp/recycle/visda-2022/checkpoints/source_only_synthzerowaste_to_zerowastev2_segformer.pth # SegFormer trained on ZeroWaste V1 + SynthWaste-aug
wget http://csr.bu.edu/ftp/recycle/visda-2022/checkpoints/uda_synthzerowaste_segformer.pth # DAFormer trained on ZeroWaste V1 + SynthWaste-aug -> ZeroWaste V2
cd ../

