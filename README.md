# P2ORM: Formulation, Inference & Application 
This repository contains the official PyTorch implementation of our paper: "Pixel-Pair Occlusion  Relationship Map (P2ORM): Formulation, Inference & Application" (ECCV 2020 Spotlight).  
It contains two parts: the estimation of occlusion boundaries with directions and the refinement on depth maps using the estimated occlusions. By using this repository, you can evaluate the pretrained models, train your own models and generate new occlusion annotations with the proposed method. For more details such as videos and introduction slides, please refer to our project webpage.

[[Project webpage]](http://imagine.enpc.fr/~qiux/P2ORM/) [[Paper]](https://arxiv.org/pdf/2007.12088.pdf) [[Supp.]](http://imagine.enpc.fr/~qiux/P2ORM/supp.pdf)

News (2021-02-01): The code for occlusion label generation is released!

News (2020-08-28): The code for training and test is released!

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

## Table of contents
* [1. Visual Results](#1-visual-results)
* [2. Installation](#2-installation)
* [3. Evaluation](#3-evaluation)
* [4. Train](#4-train)
* [5. Generate Occlusion Labels](#5-generate-occlusion-labels)

## 1. Visual Results
### 1.1 Occlusion boundary and orientation estimation
Given an image as input, the model predicts occlusion status between neighbor pixels which can be converted to occlusion boundaries with orientations (the left-side of a green arrow is the occlusion foreground). In the predictions, green color indicates the correctly estimated boundaries.
<p align="center">
<img src="https://github.com/tim885/P2ORM/blob/master/img/occ_vis_bsds-new.png" width="800px" alt="occ_vis">
</p>

### 1.2 Depth map refinement
Given an initial depth prediction and the estimated occlusions as input, the model predicts a refined depth map.
<p align="center">
<img src="https://github.com/tim885/P2ORM/blob/master/img/depth_vis_nyudv2-new.png" width="800px" alt="depth_vis">
</p>

## 2. Installation
### 2.1 Dependencies
The code has been tested on ubuntu 16.04 with a single GeForce GTX TITAN X (12GB). The main dependencies are: Python=3.6, 
Pytorch>=1.1.0 and Torchvision>=0.3.0. Please install the PyTorch version based on your CUDA version.
We recommend to utilize conda environment to install all dependencies and test the code.

```shell
# Download the repository
git clone 'https://github.com/tim885/P2ORM'
cd P2ORM

# Create python env with relevant dependencies
conda create --name P2ORM --file spec-file.txt
conda activate P2ORM
/your_conda_root/envs/P2ORM/bin/pip install tensorboardX Cython 
```

### 2.2 Data preparation
Download the relevant dataset and unzip it in folder ./data/.

#### 2.2.1 Download datasets with occlusion labels for train/test.
Download BSDS300.zip [here](https://1drv.ms/u/s!AhUxUphMG7I4jYdv3gqnp2iElQ_9iw?e=Ni7Idb).

Download InteriorNet_OR.zip [here](https://1drv.ms/u/s!AhUxUphMG7I4jYd1UPeUgMxXbi766Q?e=tK0fbD).

Download iBims1_OR.zip [here](https://1drv.ms/u/s!AhUxUphMG7I4jYdzBIOqzgHsVvP8VQ?e=gUMk53).

Download NYUv2_OR.zip [here](https://1drv.ms/u/s!AhUxUphMG7I4jYd0JZ-NwwPjuhbmZA?e=mnCPog).

#### 2.2.2 Generate new occlusion labels on a dataset (e.g., InteriorNet).
Download InteriorNet.zip [here](https://1drv.ms/u/s!AhUxUphMG7I4jaNF_Sot5YV39ja9oA?e=Nb1TSs)

N.B. The file is used only for a demonstration, please refer to [here](https://interiornet.org/) for the whole dataset.

####

## 3. Evaluation
### 3.1 Pixel-Pair Occlusion Detection
#### 3.1.1 Pretrained models
Download pretrained model for BSDSownership [here](https://1drv.ms/u/s!AhUxUphMG7I4jYc2NQVqKwBmiCROag?e=x5Emnn).

Download pretrained model for iBims1_OR [here](https://1drv.ms/u/s!AhUxUphMG7I4jYc4xePVHPNDytq2ew?e=AC74sN).

Download pretrained model for NYUv2_OR [here](https://1drv.ms/u/s!AhUxUphMG7I4jYdwlHOPtVS_CgwXdg?e=vOlT3V). 

#### 3.1.2 Test and evaluation
Use the pretrained models and test them on three datasets. The models predict occlusion status between neighbor pixels for each image and save them in the corresponding files.
```shell
# create data dir symbole link for occlusion detection and exp dir 
cd DetectOcclusion/ && ln -s ../data/ data/
mkdir DetectOcclusion/output/  && cd detect_occ/ 

# test on BSDSownership dataset and save results in ./DetectOcclusion/experiments/output/BSDSownership_pretrained/
mkdir DetectOcclusion/output/BSDSownership_pretrained/  # and put pretrained model here
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --evaluate --resume BSDSownership_pretrained/BSDSownership_epoch19.pth.tar --gpus 0

# test on iBims1_OR dataset and save results in ./DetectOcclusion/experiments/output/iBims1_OR_pretrained/
mkdir DetectOcclusion/output/iBims1OR_pretrained/  # and put pretrained model here
python train_val.py --config ../experiments/configs/ibims_order_myUnet_CCE_1.yaml --evaluate --resume iBims1OR_pretrained/iBims1OR_epoch179.pth.tar --gpus 0

# test on NYUv2_OR dataset and save results in ./DetectOcclusion/experiments/output/NYUv2_OR_pretrained/
mkdir DetectOcclusion/output/NYUv2OR_pretrained/  # and put pretrained model here
python train_val.py --config ../experiments/configs/nyuv2_order_myUnet_CCE_1.yaml --evaluate --resume NYUv2OR_pretrained/NYUv2OR_epoch74.pth.tar --gpus 0

# (Optional) we use cython to boost the speed of NMS. Run this if you can't run the test code. 
cd DetectOcclusion/lib/dataset
python setup_cython.py build_ext --inplace

# evaluate the predictions with matlab (default: BSDSownership)
# please refer to evaluation_matlab/README.md for more details
cd ../evaluation_matlab/evaluation/
matlab -nodisplay
run EvaluateOcc.m
```
The pairwise occlusion predictions (.npz files) for a dataset (e.g., iBims1OR) are stored in the following directory:
```
./DetectOcclusion/experiments/output/iBims1OR_pretrained/results_vis/test_179_ibims/res_mat/test_order_nms_pred/
```
N.B. Each .npz file contains the first channel for occlusion boundary probability (0~127) and next eight channels of occlusion relationship
w.r.t each pixel's eight neighbor pixels with label (1: occludes, -1: occluded, 0: no occlusion). The mapping from channel index to pixel neighbor is:<br/>
1 2 3<br/>
4 &nbsp;&nbsp; 5<br/>
6 7 8<br/>
```shell
# Please load the .npz file using:
occ = numpy.load('file_path')['order']
```

We also offer pairwise occlusion predictions directly to encourage using them in downstream tasks:

Download pairwise occlusion predictions for iBims-1 [here](https://1drv.ms/u/s!AhUxUphMG7I4jYdxC6qiKGoHKXB0Lg?e=eUPsYS).

Download pairwise occlusion predictions for NYUv2 [here](https://1drv.ms/u/s!AhUxUphMG7I4jYdyMeuvdCZyKE5OWw?e=Gann3f).

### 3.2 Depth Refinement with Detected Occlusion

Change to the directory: ``cd ./DepthRefine/``.

Pretrained model is saved at ``pretrained/ckpt.pth``.

To refine the depth map on iBims1_OR:
```
python refine_ibims.py --checkpoint #model_path --depth_pred_method #method_name
```
where the initial depth maps are predicted by **#method_name**, 
which should be chosen from: 
**\[
[eigen](https://arxiv.org/abs/1406.2283),
[fcrn](https://arxiv.org/abs/1606.00373), 
[fayao](https://arxiv.org/abs/1411.6387), 
[junli](https://arxiv.org/abs/1607.00730), 
[planenet](https://arxiv.org/abs/1804.06278), 
[sharpnet](https://arxiv.org/abs/1905.08598)
\]**.


To refine and evaluate depth on NYUv2_OR:
```
python refine_nyu.py --checkpoint #model_path --result_dir #refined_depths
python eval_nyu.py #refined_depths
```
the refined depths would be saved in **#refined_depths** and 
evaluation results would be logged in file **#refined_depths/eval.txt**.
The initial depth maps are predicted by:
 **\[
[eigen](https://arxiv.org/abs/1406.2283),
[laina](https://arxiv.org/abs/1606.00373), 
[dorn](https://arxiv.org/abs/1806.02446),
[sharpnet](https://arxiv.org/abs/1905.08598),
[jiao](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jianbo_Jiao_Look_Deeper_into_ECCV_2018_paper.pdf),
[vnl](https://arxiv.org/abs/1907.12209)
\]**.


## 4. Train
### 4.1 Pixel-Pair Occlusion Detection
```shell
# create data dir symbole link for occlusion detection and exp dir 
cd DetectOcclusion/ && ln -s ../data/ data/
mkdir DetectOcclusion/output/ && cd detect_occ/

# train for BSDSownership dataset
python train_val.py --config ../experiments/configs/BSDSownership_order_myUnet_CCE_1.yaml --gpus 1   

# train for iBims1_OR dataset
python train_val.py --config ../experiments/configs/ibims_order_myUnet_CCE_1.yaml --gpus 1

# train for NYUv2_OR dataset
python train_val.py --config ../experiments/configs/nyuv2_order_myUnet_CCE_1.yaml --gpus 1
```

### 4.2 Depth Refinement

Our model is trained on 10,160 images of InteriorNet_OR for 30 epochs:
```
cd ./DepthRefine/
python train.py --save_dir # save_model_path
```
The whole training procedure can be finished in ~10 hours with a single TitanX GPU.

### 5. Generate Occlusion Labels
You can generate the occlusion relationship label between immediate neighbor pixels using our method proposed in the paper. Please follow the instructions presented below:
```shell script
# create data dir symbole link  
cd DetectOcclusion/ && ln -s ../data/ data/ && cd utils

# generate occlusion labels on InteriorNet (please refer to the script for more details)
python gen_InteriornetOR_label.py
```
