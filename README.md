# RLA-Net: Recurrent Layer Aggregation

Recurrence along Depth: Deep Networks with Recurrent Layer Aggregation

This is an implementation of RLA-Net.

![RLANet](figures/rlanet.png)

## Introduction
This paper proposes a recurrent layer aggregation (RLA) module that makes use of the sequential structure of layers in a deep network to incorporate the features in all previous layers. It originates from an interpretable simplification of DenseNet, which shares the same functionality with RLA but is a bit redundant as a standalone CNN. The recurrent structure makes the proposed module very light-weighted. We show that our RLA modules are compatible with most CNNs nowadays by applying them to commonly used deep networks such as ResNets and Xception. The resulting models are tested on benchmark datasets, and the RLA module can effectively increase classification accuracy on CIFAR-10, CIFAR-100 and ImageNet.

## RLA module


<img src="figures/rla_module.png" width="350" alt="RLA_module"/><br/>

## Changelog

2021/04/06 Upload RLA-ResNet model.

## Installation

### Requirements

- Python 3.5+
- PyTorch 1.0+
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

## Quick Start

### Training
1. To train an RLA-Net using ImageNet dataset and ResNet50 as backbone with batch size = 256
```bash
python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 '/dev/shm/imagenet/'
```
2. To train an RLA-Net with a pretrained model (checkpoint)
```bash
python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50/checkpoint.pth.tar' --action 'part2' '/dev/shm/imagenet/'
```
