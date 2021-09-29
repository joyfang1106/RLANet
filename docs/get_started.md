## Get Started

### Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

### My Environment

- CUDA Version 10.2.89
- python 3.7
- pytorch 1.5.0
- torchvision 0.6.0
- numpy 1.19.2
- mmcv-full 1.2.7                 
- mmdet 2.10.0                   
- mmpycocotools 12.0.3                   


### Installation

Follow the steps in MMDetection [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md).

Since I foud that some steps in this guideline did not work well, I provide my steps here.

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n open-mmlab python=3.7 -y
    conda activate open-mmlab
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

    ```shell
    # CUDA 10.2
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
    # CUDA 10.1
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).
    
    [PyTorch history versions](https://pytorch.org/get-started/previous-versions/)


3. Install mmcv-full.

    ```shell
    # pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    # CUDA 10.2
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.5.0/index.html
    # CUDA 10.1
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
    ```
    
4. Clone the MMDetection repository.

    ```shell
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    ```

5. Install build requirements and then install MMDetection.

    ```shell
    pip install -r requirements/build.txt
    python setup.py develop  # not suggest to use pip install -v -e .
    ```
    
6. Make the symblic link to coco dataset

    ```shell
    mkdir data
    ln -s /path/to/coco data
    ```

### Prepare Datasets

#### 1. ImageNet

Download ImageNet dataset from official website in your local directory. 

```
/imagenet
    /train
    /val
```

Or put this dataset into your memory if the speed of reading and writing to the disk is too slow.

```shell
# root
mkdir /shm
chmod 777 /shm
mount -t tmpfs -o size=160G tmpfs /shm

# user
mkdir /shm/imagenet
tar -xf ~/imagenet_PP_NoTest.tar -C /shm/imagenet/ --checkpoint=100000 # 1GB per chkpt, total ~150G
```

#### 2. COCO2017

Download COCO2017 dataset [here](http://cocodataset.org/#download) for object detection and instance segmentation. 

We implement all detectors by using [MMDetection](https://github.com/open-mmlab/mmdetection) toolkit and employ the default settings. Keep the path as 

```
/mmdetection
    /data
        /coco
            /annotations
            /train2017
            /val2017
            /test2017
```

Make the symblic link to coco dataset via

```bash
mkdir data # create a folder called data in your local directory
ln -s /path/to/coco data  # make the symblic link
```
