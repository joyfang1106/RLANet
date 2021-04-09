### Train with ResNet

#### Use single node or multi node with multiple GPUs

Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training.

1. To train an RLA-Net using ImageNet dataset and ResNet50 as backbone with batch size = 256
  ```bash
  python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 {imagenet-folder with train and val folders}
  ```

  Example:
  ```bash
  python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 '/dev/shm/imagenet/'
  ```

2. To train an RLA-Net base on a checkpoint
  ```bash
  python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 --resume {path to latest checkpoint} {imagenet-folder with train and val folders}
  ```

  Example:
  ```bash
  python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50_/checkpoint.pth.tar' '/dev/shm/imagenet/'
  ```

#### Specify single GPU or multiple GPUs

1. To train an RLA-Net using 2 specified GPUs with batch size = 256
  ```bash
  CUDA_VISIBLE_DEVICES={device_ids} python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 {imagenet-folder with train and val folders}
  ```

  Example:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 '/dev/shm/imagenet/'
  ```

2. To train an RLA-Net base on a checkpoint using 2 specified GPUs
  ```bash
  CUDA_VISIBLE_DEVICES={device_ids} python train.py -a {model_name} --b {batch_size} --multiprocessing-distributed --world-size 1 --rank 0 --resume {path to latest checkpoint} {imagenet-folder with train and val folders}
  ```
  
  Example:
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50_/checkpoint.pth.tar' '/dev/shm/imagenet/'
  ```


### Testing

1. To evaluate the best model

  Example:
  ```bash
  python train.py -a rla_resnet50 --b 256 --multiprocessing-distributed --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50_/model_best.pth.tar' -e '/dev/shm/imagenet/'
  ```

2. To evaluate the best model using single specified GPU with batch size = 32

  Example:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python train.py -a rla_resnet50 --b 32 --world-size 1 --rank 0 --resume='work_dirs/rla_resnet50_/model_best.pth.tar' -e '/dev/shm/imagenet/'
  ```

3. To obtain the best Top-1 and Top-5 accuracy (the best model 'model_best.pth.tar' is selected by Top-1 acc)
  ```
  python best.py --log-dir {log_folder}
  ```
