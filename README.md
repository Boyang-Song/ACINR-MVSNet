# ACINR-MVSNet
## About 
This repository contains the official ***base*** implementation of the [paper](https://www.sciencedirect.com/science/article/pii/S0262885622001408) : "Implicit neural refinement based multi-view stereo network with adaptive correlation".  
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

### ACINR-MVSNet
This ***base*** implementation ———— ACINR-MVSNet is created by slightly revising ACINRMVSNetplus. Thus you can still use the model trained on the DTU dataset to predict depth maps for T&T for better results.
  
ACINR-MVSNetplus provided two schemes with different depth map resolutions: (1)full resolution version for DTU and BlendedMVS (2)half resolution version for T&T(large outdoor realistic scenes, full resolution brought more noise and extraneous points).  
|                | CDE | INR | EGN | Img |
|:--------------:|:---:|:---:|:---:|:---:|
|ACINR-MVSNet|1/4|1/4|1/2|1|
|ACINR-MVSNetplus<br>full resolution|1/4|1/2|1|1|
|ACINR-MVSNetplus<br>half resolution|1/4|1/2|1/2|1|
 
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
Train ACINR-MVSNetplus on `DTU` dataset, then predict depth maps and fuse them to get point clouds of `DTU` or `T&T`:
```
bash dtu3base.sh/dtu5base.sh
```
**Note** '3' or '5' in the filename means 'train_view_num'
### For BlendedMVS dataset
Train ACINR-MVSNetplus on `BlendedMVS` dataset, then validate and predict depth maps, and fuse them to get point clouds of `BlendedMVS` or `T&T`:
```
bash blend3base.sh/blend5base.sh
```
**Note** '3' or '5' in the filename means 'train_view_num'
## Metrics
### Pre-trained model
The pretrained model is in `./checkpoints/pretrained`. Uncomment the code and modify the path in the corresponding `.sh`. 
### DTU
|                | Acc. | Comp. | Overall |    Notes    |
|:--------------:|:----:|:-----:|:-------:|:-----------:|
|original<br>ACINR-MVSNet|0.306|0.364|0.335|800*576<br>train_view=3|
|**dtu3base**|0.307|0.353|0.330|800*576<br>train_view=3|
|dtu5base|0.306|0.352|0.329|800*576<br>train_view=5|
### BlendedMVS Validation
|                |   A1   |   A3   |    Notes    |
|:--------------:|:------:|:------:|:-----------:|
|original<br>ACINR-MVSNet|89.2%|97.2%|384*288<br>train_view=3|
|**blend3base**|89.6%|97.3%|384*288<br>train_view=3|
|blend5base|89.1%|97.0%|384*288<br>train_view=5|
### T&T
|                | Family | Francis | Horse | Lighthouse | M60 | Panther | Playground | Train | **F-score** |    Notes    |
|:--------------:|:------:|:-------:|:-----:|:----------:|:---:|:-------:|:----------:|:-----:|:-----------:|:-----------:|
|original<br>ACINR-MVSNet|64.83|39.07|41.64|54.59|53.62|51.17|55.45|47.79|51.02|train_view=5|
|**blend5base**|65.62|40.60|37.33|55.67|56.06|53.65|55.19|50.42|51.82|train_view=5|
|blend3base|59.61|32.85|38.26|53.35|52.67|51.08|51.66|47.24|48.34|train_view=3|
|dtu5base|67.11|46.31|43.14|56.05|58.49|53.33|55.30|47.89|53.45|train_view=5|
|dtu3base|65.18|45.04|40.97|54.29|56.90|51.78|54.56|47.19|51.99|train_view=3|
## Acknowledge
This repository is MAINLY based on these open source work: [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch), [FastMVSNet](https://github.com/svip-lab/FastMVSNet), [JIIF](https://github.com/ashawkey/jiif), [CasMVSNet_pl](https://github.com/kwea123/CasMVSNet_pl)  
  
We appreciate their great contributions to the CV community.
