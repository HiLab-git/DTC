## Dual-task Consistency
Code for this paper: Semi-supervised Medical Image Segmentation through Dual-task Consistency ([AAAI2021](https://ojs.aaai.org/index.php/AAAI/article/view/17066))
	
	@inproceedings{luo2021semi,
	  title={Semi-supervised Medical Image Segmentation through Dual-task Consistency},
	  author={Luo, Xiangde and Chen, Jieneng and Song, Tao and Wang, Guotai},
	  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
	  volume={35},
	  number={10},
	  pages={8801--8809},
	  year={2021}
	}
	
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
python train_la_dtc.py
```

4. Test the model
```
python test_LA.py
```
Our pre-trained models are saved in the model dir [DTC_model](https://github.com/Luoxd1996/DTC/tree/master/model) (both 8 labeled images and 16 labeled images), and the pretrained SASSNet and UAMT model can be download from [SASSNet_model](https://github.com/kleinzcy/SASSnet/tree/master/model) and [UA-MT_model](https://github.com/yulequan/UA-MT/tree/master/model). The other comparison method can be found in [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

## Results on the Left Atrium dataset (SOTA).
* The training set consists of 16 labeled scans and 64 unlabeled scans and the testing set includes 20 scans.

|Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|Reference|Released Date|
|---|---|---|---|---|---|---|
|[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|88.88|80.21|2.26|7.32|MICCAI2019|2019-10|
|[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|89.54|81.24|2.20|8.24|MICCAI2020|2020-07|
| [DTC](https://ojs.aaai.org/index.php/AAAI/article/view/17066)|89.42|80.98|2.10|7.32|AAAI2021|2020-09|
|[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|89.62|81.31| 2.06| 7.16|MICCAI2020|2020-10|
|[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|89.65| 81.35| 2.03| 7.04|MICCAI2020|2020-10|
|[MC-Net](https://arxiv.org/pdf/2103.02911.pdf)|90.34| 82.48| 1.77| 6.00|Arxiv|2021-03|

* The training set consists of 8 labeled scans and 72 unlabeled scans and the testing set includes 20 scans.

|Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|Reference|Released Date|
|---|---|---|---|---|---|---|
|[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|84.25|73.48|3.36|13.84|MICCAI2019|2019-10|
|[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|87.32|77.72|2.55|9.62|MICCAI2020|2020-07|
| [DTC*](https://ojs.aaai.org/index.php/AAAI/article/view/17066)|87.51|78.17|2.36|8.23|AAAI2021|2020-09|
|[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|85.54|75.12|3.77|13.29|MICCAI2020|2020-10|
|[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|85.91|75.75|3.31|12.67|MICCAI2020|2020-10|
|[MC-Net](https://arxiv.org/pdf/2103.02911.pdf)|87.71|78.31|2.18| 9.36|Arxiv|2021-03|
* Note that, * denotes the results from [MC-Net](https://arxiv.org/pdf/2103.02911.pdf) and the model has been openly available (provided by Dr. YiCheng), thanks for [Dr. Yicheng](https://ycwu1997.github.io/eli/).

## Acknowledgement
* This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap). 
* We thank Dr. Lequan Yu, M.S. Shuailin Li and Dr. Jun Ma for their elegant and efficient code base.
* More semi-supervised learning approaches for medical image segmentation have been summarized in [SSL4MIS](https://github.com/Luoxd1996/awesome-semi-supervised-learning-for-medical-image-segmentation).

