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

|Model|AP|AP_50|AP_75|BaiduDrive(models)|Extract code|GoogleDrive|
|:---- |:--:|:-------:|:-------:|:----------------:|:----------:|:---------:|
|Fast_R-CNN_resnet50_rla|38.8|59.6|42.0|[faster_rcnn_r50rla_fpn_1x_coco_388](https://pan.baidu.com/s/1Kz39oBtwNporxM5mSGD8rw)|q5c8|[faster_rcnn_r50rla_fpn_1x_coco_388](https://drive.google.com/file/d/16yqHnLT2ZZuLTcLDejyi7fsxPRuh36hN/view?usp=sharing)|
|Fast_R-CNN_ecanet50_rla|39.8|61.2|43.2|[faster_rcnn_r50rlaeca_fpn_1x_coco_398](https://pan.baidu.com/s/1UD-3nECcc0rYcQ6Fc86yDg)|f5xs|[faster_rcnn_r50rlaeca_fpn_1x_coco_398](https://drive.google.com/file/d/1oLZtRCNr0x8c6xACS41znmWKsm2SnqEw/view?usp=sharing)|
|Fast_R-CNN_resnet101_rla|41.2|61.8|44.9|[faster_rcnn_r101rla_fpn_1x_coco_412](https://pan.baidu.com/s/13Ec2jUrs7z32Z4ovRA0j0g)|0ri3|[faster_rcnn_r101rla_fpn_1x_coco_412](https://drive.google.com/file/d/15UqsMFKPSeBWnr-v7fy8Q2qidkj5_9Rj/view?usp=sharing)|
|Fast_R-CNN_ecanet101_rla|42.1|63.3|46.1|[faster_rcnn_r101rlaeca_fpn_1x_coco_421](https://pan.baidu.com/s/1ue02A9evqCbi7KFWeHyH1A)|cpug|[faster_rcnn_r101rlaeca_fpn_1x_coco_421](https://drive.google.com/file/d/1OhiVpiwohQG436ruUyV683xzmdfYV9el/view?usp=sharing)|
|RetinaNet_resnet50_rla|37.9|57.0|40.8|[retinanet_r50rla_fpn_1x_coco_379](https://pan.baidu.com/s/1u6aDamYPj4WRYzVAxTxgvA)|lahj|[retinanet_r50rla_fpn_1x_coco_379](https://drive.google.com/file/d/1sbKOUvSV0u1nj1WHSNVzSQTvB8PGIcwy/view?usp=sharing)|
|RetinaNet_ecanet50_rla|39.0|58.7|41.7|[retinanet_r50rlaeca_fpn_1x_coco_390](https://pan.baidu.com/s/17VHcUDWvW9CxYnScym7i3g)|adyd|[retinanet_r50rlaeca_fpn_1x_coco_390](https://drive.google.com/file/d/1okSs7HzBex9uB_AhKbF9qWq8rW8mEGrw/view?usp=sharing)|
|RetinaNet_resnet101_rla|40.3|59.8|43.5|[retinanet_r101rla_fpn_1x_coco_403](https://pan.baidu.com/s/14-QdA1pl4e0iV4DYfvKrFw)|p8y0|[retinanet_r101rla_fpn_1x_coco_403](https://drive.google.com/file/d/1PWKq1AiOf1f9dm_k7zUin9fcsQfKx7U-/view?usp=sharing)|
|RetinaNet_ecanet101_rla|41.5|61.6|44.4|[retinanet_r101rlaeca_fpn_1x_coco_415](https://pan.baidu.com/s/1ArVb6TR1ifwGMx3RXL6ILw)|hdqx|[retinanet_r101rlaeca_fpn_1x_coco_415](https://drive.google.com/file/d/1Hl7mhi-CAPnWR_m2reJug8hzhkokYQpa/view?usp=sharing)|
|Mask_R-CNN_resnet50_rla|39.5|60.1|43.3|[mask_rcnn_r50rla_fpn_1x_coco_395](https://pan.baidu.com/s/1FF3RJDTcABt1GjvmqCEXmQ)|j1x6|[mask_rcnn_r50rla_fpn_1x_coco_395](https://drive.google.com/file/d/1UrIzo9ZunyjwTRVm6qZmbJtfEJ2Fnqcp/view?usp=sharing)|
|Mask_R-CNN_ecanet50_rla|40.6|61.8|44.0|[mask_rcnn_r50rlaeca_fpn_1x_coco_406](https://pan.baidu.com/s/1Ne4Rb33VN5_UyFxtyzxBQw)|c08r|[mask_rcnn_r50rlaeca_fpn_1x_coco_406](https://drive.google.com/file/d/1i6J0h_5FZDg8BxvGvS1VBbVxHIxmJk4L/view?usp=sharing)|
|Mask_R-CNN_resnet101_rla|41.8|62.3|46.2|[mask_rcnn_r101rla_fpn_1x_coco_418](https://pan.baidu.com/s/1X_fLMF73vlBCb2GBurGCeQ)|8bsn|[mask_rcnn_r101rla_fpn_1x_coco_418](https://drive.google.com/file/d/1mGS2R5vx-u9KyQoK1WpyQwx-J7PI9lN2/view?usp=sharing)|
|Mask_R-CNN_ecanet101_rla|42.9|63.6|46.9|[mask_rcnn_r101rlaeca_fpn_1x_coco_429](https://pan.baidu.com/s/1zeLKIZPJGaM77nDNMnhrIA)|3kmz|[mask_rcnn_r101rlaeca_fpn_1x_coco_429](https://drive.google.com/file/d/1RKkoE8E6n1CG2BDaNuYGLzrxWY9j7XB_/view?usp=sharing)|



## Citation
     
```bibtex
@misc{zhao2021rlanet,
    title   = {Recurrence along Depth: Deep Networks with Recurrent Layer Aggregation}, 
    author  = {Jingyu Zhao, Yanwen Fang and Guodong Li},
    year    = {2021},
    eprint  = {},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```


## Questions

Please contact 'u3545683@hku.hk' or 'gladys17@hku.hk'.

