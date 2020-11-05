## Dual-task Consistency
Code for this paper: Semi-supervised Medical Image Segmentation through Dual-task Consistency ([DTC](https://arxiv.org/pdf/2009.04448.pdf))
* More details and comparison methods will be released if the paper is accepted. 
* The multi-classes **DTC** is under doing, and also will be released as we finished it.
## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python == 3.6 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch][torch_link].

[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://github.com/HiLab-git/DTC.git 
cd DTC
```
2. Put the data in [data/2018LA_Seg_Training Set](https://github.com/Luoxd1996/DTC/tree/master/data/2018LA_Seg_Training%20Set).

3. Train the model
```
cd code
python train_la_dtc.py or python train_la_dtc_v2.py
```

4. Test the model
```
python test_LA.py
```
Our best model is saved in the model dir [DTC_model](https://github.com/Luoxd1996/DTC/tree/master/model), and the pretrained SASSNet and UAMT model can be download from [SASSNet_model](https://github.com/kleinzcy/SASSnet/tree/master/model) and [UA-MT_model](https://github.com/yulequan/UA-MT/tree/master/model). Our implemented 3D version of [CCT](https://arxiv.org/abs/2003.09005) (with main decoder and three auxiliary decoders) will be updated as soon as possible, and the other comparison method can be found in [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

## Results on the Left Atrium dataset.
* The training set consists of 16 labeled scans and 64 unlabeled scans and the testing set includes 20 scans.

|Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|Reference|Released Date|
|---|---|---|---|---|---|---|
|[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|88.88|80.21|2.26|7.32|MICCAI2019|2019-10|
|[SASSNet](https://arxiv.org/pdf/2007.10732.pdf|89.54|81.24|2.20|8.24|MICCAI2020|2020-07|
|[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|89.62|81.31| 2.06| 7.16|MICCAI2020|2020-10|
[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|89.65| 81.35| 2.03| 7.04|MICCAI2020|2020-10|
|Orginal [DTC](https://arxiv.org/pdf/2009.04448.pdf)|89.42|80.98|2.10|7.32|Ours|2020-09|
|Updated DTC|**89.85**|**81.72**|**1.81**|**7.03**|Ours|2020-10|
<!--|Updated DTC (w/o NMS)|89.47|81.09|2.65|9.56|Ours|2020-10|-->
<!--|[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)(w/o NMS)|89.27|80.82|3.13|8.83|MICCAI2020|2020-07|-->

## Citation
If you find this repository is useful in your research, please consider to cite:

	@article{luo2020semi,
	  title={Semi-supervised Medical Image Segmentation through Dual-task Consistency},
	  author={Luo, Xiangde and Chen, Jieneng and Song, Tao and Chen, Yinan and Wang, Guotai and Zhang, Shaoting},
	  journal={arXiv preprint arXiv:2009.04448},
	  year={2020}
	}

## Acknowledgement
* This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap). 
* We thank Dr. Lequan Yu, M.S. Shuailin Li and Dr. Jun Ma for their elegant and efficient code base.
* More semi-supervised learning approaches for medical image segmentation have summarized in this repository [SSL4MIS](https://github.com/Luoxd1996/awesome-semi-supervised-learning-for-medical-image-segmentation).

