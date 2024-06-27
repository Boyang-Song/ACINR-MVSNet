# ACINR-MVSNet
## About 
This repository contains the official ***advanced*** implementation of the [paper](https://www.sciencedirect.com/science/article/pii/S0262885622001408) : "Implicit neural refinement based multi-view stereo network with adaptive correlation".  
If you find this project useful for your research, please cite:  
```
@article{song2022implicit,
  title={Implicit neural refinement based multi-view stereo network with adaptive correlation},
  author={Song, Boyang and Hu, Xiaoguang and Xiao, Jin and Zhang, Guofeng and Chen, Tianyou},
  journal={Image and Vision Computing},
  volume={124},
  pages={104511},
  year={2022},
  publisher={Elsevier}
}
```
**This work strongly borrows the insights from the previous MVS approaches. More details in the Acknowledge part.**
### Introduction
(1) We propose an implicit neural refinement module, introducing the Implicit Neural Representation idea into an end-to-end trainable MVS architecture. The proposed module contributes to recovering finer details, especially those in boundary areas.  
(2) We design an adaptive group-wise correlation similarity measure for multi-view cost aggregation, taking visibility into account while being efficient and memory-friendly.  
(3) We present a repeated top-down and bottom-up structure to extract context-aware features for both the coarse depth estimation and the enhanced Gauss-Newton depth refinement.  
  
*The overall pipeline can be divided into three steps: **coarse depth estimation(CDE)**, **implicit neural depth refinement(INR)**, and **enhanced Gauss-Newton depth refinement(EGN)**.*  
![](https://github.com/Boyang-Song/ACINR-MVSNet/blob/main/doc/Network%20Architecture.png)  

### ACINR-MVSNetplus
This ***advanced*** implementation ———— ACINR-MVSNetplus, mainly modified the INR module on the original basis for better comparison results. In particular, the depth-to/from-inverse_depth conversion made it possible to test the generalization ability of models trained on the DTU dataset directly on the Tanks and Temples(T&T) dataset, although the depth ranges of the two datasets are quite different.  
ACINR-MVSNetplus provided two schemes with different depth map resolutions: (1)full resolution version for DTU and BlendedMVS (2)half resolution version for T&T.  
|                | CDE | INR | EGN | Img |
|:--------------:|:---:|:---:|:---:|:---:|
|ACINR-MVSNet|1/4|1/4|1/2|1|
|ACINR-MVSNetplus<br>half resolution|1/4|1/2|1/2|1|
|ACINR-MVSNetplus<br>full resolution|1/4|1/2|1|1|
  
## How to use
### Installation
ACINRMVSNet is tested on:
- python 3.6
- CUDA 11.1
- pytorch 1.8.1
  
We only test our code under mentioned requirements.
### Data Preparation 
- Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) (also available at [BaiduYun](https://pan.baidu.com/s/1Wb9E6BWCJu4wZfwxm_t4TQ#list/path=%2F), PW: s2v2).
- Download the [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) and unzip it under the `dtu_training` folder obtained in the previous step.
- Download the pre-processed DTU testing data from [dtu-test-1200](https://drive.google.com/file/d/1rX0EXlUL4prRxrRu2DgLJv2j7-tpUD4D/view) (from [AACVP-MVSNet](https://github.com/ArthasMil/AACVP-MVSNet))
- For other datasets, please follow the practice in [MVSNet](https://github.com/YoYo000/MVSNet) by Yao Yao.
- Set root of datasets as env variables in `env.sh`.
### For DTU dataset
Train ACINR-MVSNetplus on `DTU` dataset (note that training requires a large amount of GPU memory), then predict depth maps and fuse them to get point clouds of `DTU`:
```
bash dtu.sh
```
### For BlendedMVS dataset
Train ACINR-MVSNetplus on `BlendedMVS` dataset, then validate and predict depth maps, and fuse them to get point clouds of `BlendedMVS`:
```
bash blend.sh
```
### For Tanks and Temples dataset
Train ACINR-MVSNetplus on `DTU` dataset, then predict depth maps and fuse them to get point clouds of `T&T`:
```
bash dtuTNT.sh
bash tnt.sh
```
Train ACINR-MVSNetplus on `BlendedMVS` dataset, then predict depth maps and fuse them to get point clouds of `T&T`:
```
bash blendTNT.sh
bash tnt.sh
```
## Metrics
### Pre-trained model
The pretrained model is in `./checkpoints/pretrained`. Uncomment the code and modify the path in the corresponding `.sh`. 
### DTU
|                | Acc. | Comp. | Overall |    Notes    |
|:--------------:|:----:|:-----:|:-------:|:-----------:|
|original<br>ACINR-MVSNet|0.306|0.364|0.335|800*576<br>train_view=3|
|Dmodel|0.326|0.327|0.326|1600*1152<br>train_view=3|
|DmodelTNT|0.305|0.350|0.328|800*576<br>train_view=3|
|Dmodel5TNT|0.306|0.345|0.325|800*576<br>train_view=5|
### BlendedMVS Validation
|                |   A1   |   A3   |    Notes    |
|:--------------:|:------:|:------:|:-----------:|
|original<br>ACINR-MVSNet|89.2%|97.2%|384*288<br>train_view=3|
|Bmodel|90.5%|97.6%|768*576<br>train_view=3|
|BmodelTNT|90.0%|97.5%|384*288<br>train_view=3|
|Bmodel5TNT|90.2%|97.4%|384*288<br>train_view=5|
## Acknowledge
This repository is MAINLY based on these open source work: [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch), [FastMVSNet](https://github.com/svip-lab/FastMVSNet), [JIIF](https://github.com/ashawkey/jiif), [CasMVSNet_pl](https://github.com/kwea123/CasMVSNet_pl)  
  
We appreciate their great contributions to the CV community.
