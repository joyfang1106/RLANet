# RLA-Net: Recurrent Layer Aggregation

Recurrence along Depth: Deep Networks with Recurrent Layer Aggregation

This is an implementation of RLA-Net.

![RLANet](figures/rlanet.png)

## Introduction
This paper proposes a recurrent layer aggregation (RLA) module that makes use of the sequential structure of layers in a deep network to incorporate the features in all previous layers. It originates from an interpretable simplification of DenseNet, which shares the same functionality with RLA but is a bit redundant as a standalone CNN. The recurrent structure makes the proposed module very light-weighted. We show that our RLA modules are compatible with most CNNs nowadays by applying them to commonly used deep networks such as ResNets and Xception. The resulting models are tested on benchmark datasets, and the RLA module can effectively increase classification accuracy on CIFAR-10, CIFAR-100 and ImageNet.

## RLA module

![RLA_module](figures/rla_module.png)
<img src="figures/rla_module.png" width="200" height="300" alt="RLA_module"/><br/>

## Changelog

2021/04/06 Upload RLA-ResNet model.

## Installation

### Requirements

- Python 3.5+
- PyTorch 1.0+
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [mmdetection](https://github.com/open-mmlab/mmdetection)
