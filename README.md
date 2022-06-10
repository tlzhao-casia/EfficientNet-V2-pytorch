# This repository contains my re-implementation of efficientnet-v2
[Paper](https://arxiv.org/abs/2104.00298)

## Running environment
Red Hat 4.8.5-44 with 8 A100 GPUs
torch==1.11.0
torchvision==0.12.0
numpy==1.22.4
pillow==9.1.0

## Training
* EfficientNet-S
```shell
python train.py --config configs/effnetv2_s.py
```
