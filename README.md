# P2ORM: Formulation, Inference & Application 
This repository contains the official Pytorch implementation of our paper: "Pixel-Pair Occlusion Relationship Map (P2ORM): Formulation, Inference & Application" (ECCV 2020 Spotlight). 

[[Project page]](http://imagine.enpc.fr/~qiux/P2ORM/)

<p align="center">
<img src="https://github.com/tim885/P2ORM/blob/master/img/overview.PNG" width="800px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing : 
``` 
@inproceedings{qiu2020pixel,
  title     = {Pixel-{P}air Occlusion Relationship Map ({P2ORM}): Formulation, Inference \& Application},
  author    = {Xuchong Qiu and Yang Xiao and Chaohui Wang and Renaud Marlet},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```

## Table of content
* [1. Visual Results](#1-visual-results)
* [2. Installation](#2-installation)
* [3. Evaluation](#3-evaluation)
* [4. Train](#4-train)

## 1. Visual Results
### 1.1 Occlusion boundary and orientation estimation
<p align="center">
<img src="https://github.com/tim885/P2ORM/blob/master/img/occ_vis_bsds.png" width="800px" alt="occ_vis">
</p>

### 1.2 Depth map refinement
<p align="center">
<img src="https://github.com/tim885/P2ORM/blob/master/img/depth_vis_nyudv2.png" width="800px" alt="depth_vis">
</p>

## 2. Installation
### 2.1 Dependencies
The code has been tested on ubuntu 16.04 with a single GeForce GTX TITAN X (12GB). The main dependencies are: Python=3.6, 
Pytorch>=1.1.0 and Torchvision>=0.3.0. Please install Pytorch version based on your CUDA version.

We recommend to utilize conda environment to install all dependencies and test the code.

```shell
# Download the repository
git clone 'https://github.com/tim885/P2ORM'
cd P2ORM

# Create python env with relevant packages
conda create --name P2ORM --file spec-file.txt
source activate P2ORM
```

### 2.2 Data preparation
#### 2.2.1 Download BSDSownership dataset
Download BSDS300.zip [here](https://1drv.ms/u/s!AhUxUphMG7I4jYcz1vGKyyA3IVTnwQ?e=qqxpxP) and unzip it in folder data/.

## 3. Evaluation
### 3.1 Pixel-Pair Occlusion Detection
#### 3.1.1 Pretrained models
Download pretrained model on BSDSownership [here](https://1drv.ms/u/s!AhUxUphMG7I4jYc2NQVqKwBmiCROag?e=x5Emnn)

#### 3.1.2 Test and evaluation
```shell
# create data dir symbole link for occlusion detection and exp dir 
cd DetectOcclusion/ && ln -s ../data/ data/
mkdir DetectOcclusion/output/  && cd detect_occ/ 

# test on BSDSownership dataset
mkdir DetectOcclusion/output/BSDSownership_pretrained/  # and put pretrained model here
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --evaluate --resume BSDSownership_pretrained/BSDSownership_epoch19.pth.tar --gpus 0

# do non-maximum suppression on predictions and evaluate (default: BSDSownership)
# please refer to evaluation_matlab/README.md for more details
cd ../evaluation_matlab/evaluation/
matlab -nodisplay
run EvaluateOcc.m   
```

## 4. Train
### 4.1 Pixel-Pair Occlusion Detection
```shell
# create data dir symbole link for occlusion detection and exp dir 
cd DetectOcclusion/ && ln -s ../data/ data/
mkdir DetectOcclusion/output/ && cd detect_occ/

# train on BSDSownership dataset
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --gpus 1   
```
