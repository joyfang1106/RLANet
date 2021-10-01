# RLA-Net: Recurrent Layer Aggregation

Recurrence along Depth: Deep Networks with Recurrent Layer Aggregation

This is an implementation of RLA-Net (accept by NeurIPS-2021, [paper](url)).

![RLANet](figures/rlanet.png)

## Introduction
This paper introduces a concept of layer aggregation to describe how information from previous layers can be reused to better extract features at the current layer. 
While DenseNet is a typical example of the layer aggregation mechanism, its redundancy has been commonly criticized in the literature. 
This motivates us to propose a very light-weighted module, called recurrent layer aggregation (RLA), by making use of the sequential structure of layers in a deep CNN. 
Our RLA module is compatible with many mainstream deep CNNs, including ResNets, Xception and MobileNetV2, and its effectiveness is verified by our extensive experiments on image classification, object detection and instance segmentation tasks. 
Specifically, improvements can be uniformly observed on CIFAR, ImageNet and MS COCO datasets, and the corresponding RLA-Nets can surprisingly boost the performances by 2-3\% on the object detection task. 
This evidences the power of our RLA module in helping main CNNs better learn structural information in images.

## Citation

    @InProceedings{zhao2021rla,
       title={Recurrence along Depth: Deep Networks with Recurrent Layer Aggregation},
       author={Jingyu Zhao, Yanwen Fang and Guodong Li},
       booktitle = {},
       year={2021}
     }

## RLA module

<img src="figures/rla_module.png" width="300" alt="RLA_module"/><br/>


## Changelog

- 2021/04/06 Upload RLA-ResNet model.
- 2021/04/16 Upload RLA-MobileNetV2 (depthwise separable conv version) model.
- 2021/09/29 Upload all the ablation study on ImageNet.
- 2021/09/30 Upload mmdetection files.
- 2021/10/01 Upload pretrained weights.

## Installation

### Requirements

- Python 3.5+
- PyTorch 1.0+
- [thop](https://github.com/Lyken17/pytorch-OpCounter)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

### Our environments

- OS: Linux Red Hat 4.8.5
- CUDA: 10.2
- Toolkit: Python 3.8.5, PyTorch 1.7.0, torchvision 0.8.1
- GPU: Tesla V100

Please refer to [get_started.md](docs/get_started.md) for more details about installation.


## Quick Start

### Train with ResNet

#### - Use single node or multi node with multiple GPUs

Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.

  ```bash
  python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 {imagenet-folder with train and val folders}
  ```

#### - Specify single GPU or multiple GPUs

  ```bash
  CUDA_VISIBLE_DEVICES={device_ids} python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 {imagenet-folder with train and val folders}
  ```

### Testing

To evaluate the best model

  ```bash
  python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 --resume {path to the best model} -e {imagenet-folder with train and val folders}
  ```

### Visualizing the training result

To generate acc_plot, loss_plot
  ```
  python eval_visual.py --log-dir {log_folder}
  ```
  
### Train with MobileNet_v2

It is same with above ResNet replace `train.py` by `train_light.py`.


### Compute the parameters and FLOPs

If you have install [thop](https://github.com/Lyken17/pytorch-OpCounter), you can `paras_flops.py` to compute the parameters and FLOPs of our models. The usage is below:
```
python paras_flops.py -a {model_name}
```

More examples are shown in [examples.md](docs/examples.md).

### MMDetection

After installing MMDetection (see [get_started.md](docs/get_started.md)), then do the following steps:

- put the file resnet_rla.py in the folder './mmdetection/mmdet/models/backbones/', and do not forget to import the model in the __init__.py file.
- put the config files (e.g. faster_rcnn_r50rla_fpn.py) in the folder './mmdetection/configs/_base_/models/'
- put the config files (e.g. faster_rcnn_r50rla_fpn_1x_coco.py) in the folder './mmdetection/configs/faster_rcnn'

Note that the config files of the latest version of MMDetection are a little different, please modify the config files according to the latest format.



## Experiments

### ImageNet
|Model|Param.|FLOPs|Top-1 err.(%)|Top-5 err.(%)|BaiduDrive(models)|Extract code|GoogleDrive|
|:--- |:----:|:---:|:------:|:------:|:----------------:|:----------:|:---------:|
|RLA-ResNet50|24.67M|4.17G|22.83|6.58|[resnet50_rla_2283](https://pan.baidu.com/s/1GrNxNariVpb9S5EUFW1eng)|5lf1|[resnet50_rla_2283](https://drive.google.com/file/d/1cetP1SdOiwznLxlBUaHG8Q8c4RIwToWW/view?usp=sharing)|
|RLA-ECANet50|24.67M|4.18G|22.15|6.11|[ecanet50_rla_2215](https://pan.baidu.com/s/1B5wVN4s_WVVq8nGiiOVncA)|xrfo|[ecanet50_rla_2215](https://drive.google.com/file/d/173qoDPGe2q5l7CKVm54-_xJtg3_UFRR-/view?usp=sharing)|
|RLA-ResNet101|42.92M|7.79G|21.48|5.80|[resnet101_rla_2148](https://pan.baidu.com/s/1sZQlAU4ksIjnOUg4iSSO-Q)|zrv5|[resnet101_rla_2148](https://drive.google.com/file/d/1V9Iv0KbN1O92ll8rcf45kLkD9EOA9VCE/view?usp=sharing)|
|RLA-ECANet101|42.92M|7.80G|21.00|5.51|[ecanet101_rla_2100](https://pan.baidu.com/s/1ILfQ8pK1WdnAxSWb5X88PQ)|vhpy|[ecanet101_rla_2100](https://drive.google.com/file/d/1QMR_yf0RYugpJosCSo0uNRBmfl2e7cGa/view?usp=sharing)|
|RLA-MobileNetV2|3.46M|351.8M|27.62|9.18|[dsrla_mobilenetv2_k32_2762](https://pan.baidu.com/s/135Id3juTsj0IAo0jSKooxw)|g1pm|[dsrla_mobilenetv2_k32_2762](https://drive.google.com/file/d/1yg9hsACBHZFT5R8s95igJTyaQ5iYKklV/view?usp=sharing)|
|RLA-ECA-MobileNetV2|3.46M|352.4M|27.07|8.89|[dsrla_mobilenetv2_k32_eca_2707](https://pan.baidu.com/s/1YVN5Qze51HI9D6nNEb7iPA)|9orl|[dsrla_mobilenetv2_k32_eca_2707](https://drive.google.com/file/d/1JdEkJg9_IOnJsHWKVPQ-4YHBVabiNJXD/view?usp=sharing)|



### COCO 2017


## Questions

Please contact 'u3545683@hku.hk' or 'gladys17@hku.hk'.

