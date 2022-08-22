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
| name               | supernet | search            | mAcc | #params | FLOPs |
|--------------------|----------|-------------------|------|---------|-------|
| Auto-Points(base)  | [model](https://drive.google.com/file/d/1WNmPsZUo_BEtY2_gYY1wzQ1b6uqv2BoL/view?usp=drivesdk)    | [model](https://drive.google.com/file/d/1rg5sYrazU_ztJjbFrZcUeJBqK4f-tblc/view?usp=drivesdk)\|[log](https://drive.google.com/file/d/10OVf0BN1kjalizdKUxG3Rrm8AnE1rAp_/view?usp=drivesdk)\|[mutable](https://drive.google.com/file/d/1ZbRpPbId3-EsJEJ6QhND08AzNJMWsyIX/view?usp=drivesdk) | 93.0 | 1.9M    | 2.9G  |
| Auto-Points(large) | [model](https://drive.google.com/file/d/1jRHlj4FFrMEPkCvHRt0m9ZfUuU7yy_tC/view?usp=drivesdk)    | [model](https://drive.google.com/file/d/1t35XSvxZBVABJ_KqcdCalA8c9bblhc43/view?usp=drivesdk)\|[log](https://drive.google.com/file/d/1ge2O_c6fN1O4XWUUg6-gEH7mGKBqKss0/view?usp=drivesdk)\|[mutable](https://drive.google.com/file/d/17E-hH2ypIxZzgm1BBugoW-IPhZhMlBu2/view?usp=drivesdk) | 93.8 | 3.8M    | 4.6G  |

#### S3DIS(Area 5)
| name               | supernet | search            | mIoU | #params | FLOPs |
|--------------------|----------|-------------------|------|---------|-------|
| Auto-Points(base)  | [model](https://drive.google.com/file/d/1ZX-4TrugHutvsyi_P6bQ6S_7ZN7aoKQZ/view?usp=drivesdk)    | [model](https://drive.google.com/file/d/1QudxaX9rDzCJ9TCxG2yjSV_NR4jY8mGL/view?usp=drivesdk)\|[log](https://drive.google.com/file/d/12h4CylD1CVXcFJ6WJMtTUIiEQllbR-li/view?usp=drivesdk)\|[mutable](https://drive.google.com/file/d/1dsqNFZ9OaqKNJ5dJSBbk-KenYc_UJhIb/view?usp=drivesdk) | 63.2 | 0.8M    | 1.0G  |
| Auto-Points(large) | [model](https://drive.google.com/file/d/1J7jqMQYVXnyNrsppgVOW52kuofYqYS_M/view?usp=drivesdk)    | [model](https://drive.google.com/file/d/1CuVgHDH3v1ebYG3Wup-E9K5UVrk9nlzP/view?usp=drivesdk)\|[log](https://drive.google.com/file/d/1nAFxd2xMINiYY3ImAgGKgSlYdILFYb5m/view?usp=drivesdk)\|[mutable](https://drive.google.com/file/d/18B2G8p_xWE5C8y3FX9agC5M2S4FGN1aJ/view?usp=drivesdk) | 67.7 | 16.9M   | 6.8G  |

#### SUNRGB-D
| name               | supernet | search            | mAP@0.25 | #params | FLOPs |
|--------------------|----------|-------------------|----------|---------|-------|
| Auto-Points(base)  | [model](https://drive.google.com/file/d/1JeLtMeAaMXpVklRKhDz758bEtOW3XcP7/view?usp=drivesdk)    | [model](https://drive.google.com/file/d/1E3UJ-KZHe6HD-BQnEw1Zh4GoKShVUkg8/view?usp=drivesdk)|[log](https://drive.google.com/file/d/1MdfUFalOu4Q4D3JqaFJZCu2NKU3abC5I/view?usp=drivesdk)|[mutable](https://drive.google.com/file/d/1tnqbMGXOvJFy1epOnaYubeKdDhr7yfwn/view?usp=drivesdk) | 61.9     | 1.0M    | 5.5G  |
| Auto-Points(large) | [model](https://drive.google.com/file/d/18_4XuWrgvmsyhPBV46liGqH8ulAzyA2S/view?usp=drivesdk)   | [model](https://drive.google.com/file/d/1J4UTAaUqOjTX0l6PJlvwcu5X-gJKxHxG/view?usp=drivesdk)|[log](https://drive.google.com/file/d/1CFwd_qBcEqAn3vcI5HPtLDQm-OhDY8j6/view?usp=drivesdk)|[mutable](https://drive.google.com/file/d/1GPZ7wJ_Azn7vjkaZvfT891z3QGwkhokD/view?usp=drivesdk) | 63.2     | 13.7M   | 25.6G |



