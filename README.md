# RLA-Net: Recurrent Layer Aggregation

Recurrence along Depth: Deep Networks with Recurrent Layer Aggregation

This is an implementation of RLA-Net.

![RLANet](figures/rlanet.png)

## Introduction
This paper proposes a recurrent layer aggregation (RLA) module that makes use of the sequential structure of layers in a deep network to incorporate the features in all previous layers. It originates from an interpretable simplification of DenseNet, which shares the same functionality with RLA but is a bit redundant as a standalone CNN. The recurrent structure makes the proposed module very light-weighted. We show that our RLA modules are compatible with most CNNs nowadays by applying them to commonly used deep networks such as ResNets and Xception. The resulting models are tested on benchmark datasets, and the RLA module can effectively increase classification accuracy on CIFAR-10, CIFAR-100 and ImageNet.

## RLA module


<img src="figures/rla_module.png" width="350" alt="RLA_module"/><br/>

## Changelog

- 2021/04/06 Upload RLA-ResNet model.
- TO DO: MobileNetV2 model.

## Installation

### Requirements

- Python 3.5+
- PyTorch 1.0+
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

Please refer to [get_started.md](docs/get_started.md) for more details about installation.


## Quick Start

### Training

#### Use single node or multi node with multiple GPUs

Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.

1. To train an RLA-Net using ImageNet dataset and ResNet50 as backbone with batch size = 256
  ```bash
  python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 '/dev/shm/imagenet/'
  ```

2. To train an RLA-Net base on a checkpoint
  ```bash
  python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50/checkpoint.pth.tar' --action 'part2' '/dev/shm/imagenet/'
  ```

#### Specify single GPU or multiple GPUs

1. To train an RLA-Net using 2 specified GPUs with batch size = 256
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 '/dev/shm/imagenet/'
  ```

2. To train an RLA-Net base on a checkpoint using 2 specified GPUs
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50/checkpoint.pth.tar' --action 'part2' '/dev/shm/imagenet/'
  ```

### Testing

1. To evaluate the best model
  ```bash
  python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50/model_best.pth.tar' -e '/dev/shm/imagenet/'
  ```

2. To evaluate the best model using single specified GPU with batch size = 32
  ```bash
  CUDA_VISIBLE_DEVICES=0 python train.py -a rla_resnet50 --b 32 --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50/model_best.pth.tar' -e '/dev/shm/imagenet/'
  ```

3. To obtain the best Top-1 and Top-5 accuracy (the best model 'model_best.pth.tar' is selected by Top-1 acc)
  ```bash
  python best.py --log-dir rla_resnet50
  ```
