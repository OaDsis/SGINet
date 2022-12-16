# SGINet: Toward Sufficient Interaction Between Single Image Deraining and Semantic Segmentation (ACM MM 2022)
Yanyan Wei, Zhao Zhang, Huan Zheng, Richang Hong, Yi Yang, Meng Wang

### Update
2022.10.26: Fix the code and re-upload the datasets, if you have downloaded our datasets, please re-download the **Cityscapes_syn (200mm)** file in data link, and replace the previous one. If you have not download 'resnet101_v2.pth', please down it and put it in "./SGINet/src/initmodel".


### Abstract
Data-driven single image deraining (SID) models have achieved greater progress by simulations, but there is still a large gap between current deraining performance and practical high-level applications, since high-level semantic information is usually neglected in current studies. Although few studies jointly considered high-level tasks (e.g., segmentation) to enable the model to learn more high-level information, there are two obvious shortcomings. First, they require the segmentation labels for training, limiting their operations on other datasets without high-level labels. Second, high- and low-level information are not fully interacted, hence having limited improvement in both deraining and segmentation tasks. In this paper, we propose a Semantic Guided Interactive Network (SGINet), which considers the sufficient interaction between SID and semantic segmentation using a three-stage deraining manner, i.e., coarse deraining, semantic information extraction, and semantics guided deraining. Specifically, a Full Resolution Module (FRM) without down-/up-sampling is proposed to predict the coarse deraining images without context damage. Then, a Segmentation Extracting Module (SEM) is designed to extract accurate semantic information. We also develop a novel contrastive semantic discovery (CSD) loss, which can instruct the process of semantic segmentation without real semantic segmentation labels. Finally, a triple-direction U-net-based Semantic Interaction Module (SIM) takes advantage of the coarse deraining images and semantic information for fully interacting low-level with high-level tasks. Extensive simulations on the newly- constructed complex datasets Cityscapes_syn and Cityscapes_real demonstrated that our model could obtain more promising results. Overall, our SGINet achieved SOTA deraining and segmentation performance in both simulation and real-scenario data, compared with other representative SID methods. 

![image](https://github.com/OaDsis/SGINet/blob/main/figures/model.png)
![image](https://github.com/OaDsis/SGINet/blob/main/figures/table1.png)
![image](https://github.com/OaDsis/SGINet/blob/main/figures/table3.png)
![image](https://github.com/OaDsis/SGINet/blob/main/figures/illustration.png)

### Requirements
- Python 3.8.10
- torch 1.10.0+cu113
- torchvision 0.11.1+cu113

### Datasets
- Cityscapes_syn, include two types rain speed, i.e, 100mm and 200mm. There are 2,975, 1,525, and 500 image pairs in trainset, testset, and valset, respectively.
- Cityscapes_real

You can download above datasets from [Baidu Drive](https://pan.baidu.com/s/14Qj0ZX-SOcbKZFkqq12gcg) (Key：ky93).

### Model
- train_epoch_200_psp101.pth
- resnet101_v2.pth

You need download above pre-trained models of PSPNet101 from [Baidu Drive](https://pan.baidu.com/s/14Qj0ZX-SOcbKZFkqq12gcg) (Key：ky93) and put it in "./SGINet/src/initmodel/train_epoch_200_psp101.pth" and "./SGINet/src/initmodel/resnet101_v2.pth".

### Usage
#### Prepare dataset:
Taking training Cityscapes_syn (200mm) as an example. Download them (including training, testing, and validation set) and put them into the folder "./data", then the content is just like:

"./SGINet/data/train/200mm/rain/norain-***.png"

"./SGINet/data/train/200mm/norain/norain-***.png"

"./SGINet/data/test/200mm/rain/norain-***.png"

"./SGINet/data/test/200mm/norain/norain-***.png"

"./SGINet/data/val/200mm/rain/norain-***.png"

"./SGINet/data/val/200mm/norain/norain-***.png"
#### Train:
```
python main.py --save 200mm --model sginet --scale 2 --epochs 200 --batch_size 4 --patch_size 512 --data_train RainHeavy --n_threads 0 --data_test RainHeavyTest --stage 1 --lr_decay 25 --gamma 0.2 --num_M 32 --num_Z 32 --data_range 1-2975/1-1525 --loss 1*SSIM+1*PER+1*SEG+0.01*MSE
```
#### Test:
```
python test.py --model sginet --model_dir ../experiment/200mm/model --data_path ../data/test/200mm/rain --save_path ../experiment/200mm/results
```
### Citation
Please cite our paper if you find the code useful for your research.
```
@inproceedings{wei2022sginet,
  title={SGINet: Toward Sufficient Interaction Between Single Image Deraining and Semantic Segmentation},
  author={Wei, Yanyan and Zhang, Zhao and Zheng, Huan and Hong, Richang and Yang, Yi and Wang, Meng},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1--9},
  year={2022},
  publisher={ACM}
}
```

### Contact
Thanks for your attention. If you have any questions, please contact my email: weiyanyan@mail.hfut.edu.cn. 
