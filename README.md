# DTC
Code for work: Semi-supervised Medical Image Segmentation through Dual-task Consistency

This code is origin from [UA-MT](https://github.com/yulequan/UA-MT) and [SASSNet](https://github.com/kleinzcy/SASSnet)

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
