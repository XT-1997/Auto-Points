## Auto-Points
Official PyTorch implementation of Auto-Points, for the following paper:
> **Learning to Analysis Point Cloud with Neural Architecture Search**<br>

### Installation
- Python 3.6+
- PyTorch 1.10.1
- CUDA 10.1+ 
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate it.

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```

On CPU platforms:

```shell
conda install pytorch torchvision cpuonly -c pytorch
```

# Installation

We recommend that users follow our best practices to install MMDetection3D. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

## Best Practices
Assuming that you already have CUDA 11.0 installed, here is a full script for quick installation of MMDetection3D with conda.
Otherwise, you should refer to the step-by-step installation instructions in the next section.

```shell
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

**Step 1.** Install [MMDetection](https://github.com/open-mmlab/mmdetection).


```shell
pip install mmdet
```

Optionally, you could also build MMDetection from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.0  # switch to v2.24.0 branch
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**Step 2.** Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

```shell
pip install mmsegmentation
```

Optionally, you could also build MMSegmentation from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.20.0  # switch to v0.20.0 branch
pip install -e .  # or "python setup.py develop"
```

**Step 3.** Clone the MMDetection3D repository.

```shell
git clone https://github.com/XT-1997/RVS-VoteNet.git
cd RVS-VoteNet/mmdetection3d
```

**Step 4.** Install build requirements and then install MMDetection3D.

```shell
pip install -v -e .  # or "python setup.py develop"
```
### Getting Started

Please see [getting_started.md](docs/getting_started.md) for basic usage examples.
We follow the `mmdetection3d` data preparation protocol described in [mobilenet40](data/mobilenet40), [sunrgbd](data/sunrgbd), and [s3dis](data/s3dis).
The only difference is that we [do not sample](tools/data_converter/sunrgbd_data_utils.py#L143) 50,000 points from each point cloud in `SUN RGB-D`, using all points.

*Training**

To start training, run [slurm_train](tools/slurm_train.sh) with `rvs_votenet` [configs](mmdetection3d/configs/rvs_votenet):
```shell
cd mmdetection3d
bash tools/slurm_train.sh configs/rvs_votenet/rvs-votenet_16x8_sunrgbd-3d-10class.py --work_dir
```

**Testing**

Test pre-trained model using [slurm_test](tools/slurm_test.sh) with `rvs_votenet` [configs](mmdetection3d/configs/rvs_votenet):
```shell
bash tools/slurm_test.sh configs/rvs_votenet/rvs-votenet_16x8_sunrgbd-3d-10class.py \
    work_dirs/rvs_votenet_sunrgbd-3d-10class/latest.pth --work_dir --eval mAP
```
### Models
We provide the training logs & pretrained models through Google Drive.

#### MobileNet40
