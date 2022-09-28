# SGINet: Toward Sufficient Interaction Between Single Image Deraining and Semantic Segmentation (ACM MM 2022)
Yanyan Wei, Zhao Zhang, Huan Zheng, Richang Hong, Yi Yang, Meng Wang

### Abstract
Data-driven single image deraining (SID) models have achieved greater progress by simulations, but there is still a large gap between current deraining performance and practical high-level applications, since high-level semantic information is usually neglected in current studies. Although few studies jointly considered high-level tasks (e.g., segmentation) to enable the model to learn more high-level information, there are two obvious shortcomings. First, they require the segmentation labels for training, limiting their operations on other datasets without high-level labels. Second, high- and low-level information are not fully interacted, hence having limited improvement in both deraining and segmentation tasks. In this paper, we propose a Semantic Guided Interactive Network (SGINet), which considers the sufficient interaction between SID and semantic segmentation using a three-stage deraining manner, i.e., coarse deraining, semantic information extraction, and semantics guided deraining. Specifically, a Full Resolution Module (FRM) without down-/up-sampling is proposed to predict the coarse deraining images without context damage. Then, a Segmentation Extracting Module (SEM) is designed to extract accurate semantic information. We also develop a novel contrastive semantic discovery (CSD) loss, which can instruct the process of semantic segmentation without real semantic segmentation labels. Finally, a triple-direction U-net-based Semantic Interaction Module (SIM) takes advantage of the coarse deraining images and semantic information for fully interacting low-level with high-level tasks. Extensive simulations on the newly- constructed complex datasets Cityscapes_syn and Cityscapes_real} demonstrated that our model could obtain more promising results. Overall, our SGINet achieved SOTA deraining and segmentation performance in both simulation and real-scenario data, compared with other representative SID methods. 

![image](https://github.com/OaDsis/DerainCycleGAN/blob/main/figures/model.png)

### Requirements
- python 3.6.10
- torch 1.4.0
- torchvision 0.5.0

### Datasets
- Cityscapes_syn
- Cityscapes_real

You can download above datasets from [here](https://github.com/hongwang01/Video-and-Single-Image-Deraining#datasets-and-discriptions)

### Usage
#### Prepare dataset:
Taking training Rain100L as an example. Download Rain100L (including training set and testing set) and put them into the folder "./datasets", then the content is just like:

"./datasets/rainy_Rain100L/trainA/rain-***.png"

"./datasets/rainy_Rain100L/trainB/norain-***.png"

"./datasets/test_rain100L/trainA/rain-***.png"

"./datasets/test_rain100L/trainB/norain-***.png"
#### Train:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --train_path ../datasets/rainy_Rain100L --val_path ../datasets/test_rain100L --name TEST
```
#### Test:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test.py --test_path ../datasets --name TEST --resume ../results/TEST/net_best_*****.pth --mode 1
```
#### Generate Rain Images
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 auto.py --auto_path ../datasets --name Rain100L_new --resume ../results/TEST/net_best_*****.pth --mode 0 --a2b 0
```
### Citation
Please cite our paper if you find the code useful for your research.
```
@article{wei2021deraincyclegan,
  title={Deraincyclegan: Rain attentive cyclegan for single image deraining and rainmaking},
  author={Wei, Yanyan and Zhang, Zhao and Wang, Yang and Xu, Mingliang and Yang, Yi and Yan, Shuicheng and Wang, Meng},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={4788--4801},
  year={2021},
  publisher={IEEE}
}
```

### Contact
Thanks for your attention. If you have any questions, please contact my email: weiyanyan@mail.hfut.edu.cn. 
