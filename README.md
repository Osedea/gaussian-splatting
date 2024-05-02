# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

This repository contains the official authors implementation associated with the paper "3D Gaussian Splatting for Real-Time Radiance Field Rendering", which can be found [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). The code is based of the authors [repository](https://github.com/graphdeco-inria/gaussian-splatting). You may want to refer to it for additional ressources. 

# Development Setup

## Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- Docker 

If you do not have access to such compute ressources, look into [Modal](#modal) 
section below to train models remotely. 

## Step-by-Step Setup

### 1. Clone Repository. 

The repository contains submodules - make sur you have them properly cloned and
initialized. 

```shell
git clone git@github.com:Osedea/gaussian-splatting.git --recursive
cd gaussian-splatting/
```

### 2. Build the Docker Image.

We provide a Docker image to help setup. Build the container to install 
dependencies. 

```shell
docker build -t gaussian_splatting . 
```

### 3. Run a Docker Container.

For development, we suggest running the Docker conrtainer in interactive mode. 
To do so, you must mount the proper resources. 

```shell
docker run -it --gpus all --rm -v $(pwd):/workspace/ gaussian_splatting 
```

### 4. Activate the Conda Environment.

```shell
conda activate gaussian_splatting
```

### 5. [Optional] Test your Environment. 

To test that your environment is functional and can access GPU resources 
properly, run the following command. The output should be `True`.

```shell
python -c "import torch; print(torch.cuda.is_available());"
```

If this test fails, there is likely a compatibility issue with the CUDA toolkit 
and nvidia drivers installed locally and the one in your docker environment. 
You might want to onsider changing the base cuda image. 

## Train your First Gaussian Splat Model!

### 1. Prepare Dataset

To train your first model, you may either want to start with a series of images
in a folder called `input`. 

```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```

If you have a video, you may use this script to extract frames and output images
in the expected format. 

```shell
python scripts/extract_video_frames.py -v <path to .mp4 file> -k <frame_step_size>
```

#### 1.1. Pose Estimation

The process of training Gaussian splats models requires a dataset of image and 
camera pose pairs. We provide two options for camera pose estimation:

##### Option 1 - Prior Pose Estimation 

1. The first option consists in estimating poses with the Structure from Motion 
algorithm (COLMAP software). This option is recommended for high resolution images 
of fairly complex scenes with good keypoints. 

```shell
python scripts/convert.py -s <path to images location>
```

The script generates a dataset in the expected format for training and 
evaluation. 
```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

##### Option 2 - Pose Free Training

2. The second option consists in learning new camera poses during training. This 
option requires sequential images, when the transformation between frame T and 
frame T + 1 can be learned as an affine transformation. This option is recommended
when Option 1 fails to extract enough camera poses for reconstruction - this 
typically occurs when image resolution is low, or when the images are textureless. 

### 2. Train Gaussians

By default, the training script expects the dataset to point to a folder with 
images and camera pose estimations. To run the pose free algorithm and learn 
camera poses during training, pass the `--pose-free` flag to training script.

```shell
python scripts/train.py -s <path to dataset>
```

# Viewer

To view the resulting model, we recommend using this web viewer:
[https://antimatter15.com/splat/](https://antimatter15.com/splat/). You can 
simply drag and drop your point_cloud.ply file to view the 3D gaussian splat 
model.


# Modal 

If you do not have access to enough computation power, the alternative is to run
training remotely on the cloud GPU provider [Modal](https://modal.com/). 

First sign up using your Github account and navigate to your settings to
generate an API token. 

### 0. Setup on Modal

First, install Modal CLI in a virtual environment via pip, and setup your API
credentials. 

```shell
pip install modal
modal token set --token-id <id> --token-secret <secret>
```

### 1. Push your Dataset

Push your COLMAP dataset into the `data` network file system so that Modal can
access it during training.

```shell
modal nfs put data <local_path> <remote_path>
```

### 2. Run Remote Training

Launch remote training. 

```shell
modal run scripts/train_modal.py
```

The first time the script is launched, a Modal image has to be built, which 
may require a little more time. For former trainings, the image will be cached
to streamline the process. 

Expect between 30 minutes to an hour for a 30,000 iterations training to 
complete. 

### 3. Fetch the Model

When training is completed, a folder is committed in the `model_registry` 
volume. Navigate the registry to locate the resulting `point_cloud.ply` and 
fetch it locally to visualize results in the web viewer. 

```shell
modal volume ls model_registry
modal volume get model_registry <remote_path> <local_path>
```
