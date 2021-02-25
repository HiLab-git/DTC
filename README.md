## Dual-task Consistency
Code for this paper: Semi-supervised Medical Image Segmentation through Dual-task Consistency ([AAAI2021](https://arxiv.org/pdf/2009.04448.pdf))
	
	@article{luo2021semi,
	  title={Semi-Supervised Medical Image Segmentation through Dual-task Consistency},
	  author={Luo, Xiangde and Chen, Jieneng and Song, Tao and Chen, Yinan and Wang, Guotai and Zhang, Shaoting},
	  journal={AAAI Conference on Artificial Intelligence},
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
Our pre-trained models are saved in the model dir [DTC_model](https://github.com/Luoxd1996/DTC/tree/master/model), and the pretrained SASSNet and UAMT model can be download from [SASSNet_model](https://github.com/kleinzcy/SASSnet/tree/master/model) and [UA-MT_model](https://github.com/yulequan/UA-MT/tree/master/model). The other comparison method can be found in [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

## Acknowledgement
* This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap). 
* We thank Dr. Lequan Yu, M.S. Shuailin Li and Dr. Jun Ma for their elegant and efficient code base.
* More semi-supervised learning approaches for medical image segmentation have been summarized in [SSL4MIS](https://github.com/Luoxd1996/awesome-semi-supervised-learning-for-medical-image-segmentation).

