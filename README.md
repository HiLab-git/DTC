# DTC
Code for work: Semi-supervised Medical Image Segmentation through Dual-task Consistency ([DTC](https://arxiv.org/pdf/2009.04448.pdf))

This code is adapted from [UA-MT](https://github.com/yulequan/UA-MT), [SASSNet](https://github.com/kleinzcy/SASSnet), [SegWithDistMap](https://github.com/JunMa11/SegWithDistMap), 

We thank Dr. Lequan Yu, M.S. Shuailin Li and Dr. Jun Ma for their elegant and efficient code base. (Apologizing to Shuailin Li since I made a mistake of his degree.)

More details and comparison methods will be released if the paper is accepted. 
# Usage

1. Clone the repo:
```
git clone https://github.com/Luoxd1996/DTC.git 
cd DTC
```
2. Put the data in `data/2018LA_Seg_Training Set`.

3. Train the model
```
cd code
python train_la_dtc.py or python train_la_dtc_v2.py
```

4. Test the model
```
python test_LA.py
```
Our best model are saved in model dir.
