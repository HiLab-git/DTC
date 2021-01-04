## Results on the Left Atrium dataset (SOTA).
* The training set consists of 16 labeled scans and 64 unlabeled scans and the testing set includes 20 scans.

|Methods|DICE (%) | Jaccard (%) | ASD (voxel) | 95HD (voxel)|Reference|Released Date|
|---|---|---|---|---|---|---|
|[UAMT](https://arxiv.org/pdf/1907.07034.pdf)|88.88|80.21|2.26|7.32|MICCAI2019|2019-10|
|[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)|89.54|81.24|2.20|8.24|MICCAI2020|2020-07|
|Original [DTC](https://arxiv.org/pdf/2009.04448.pdf)|89.42|80.98|2.10|7.32|Arxiv|2020-09|
|[LG-ER-MT](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|89.62|81.31| 2.06| 7.16|MICCAI2020|2020-10|
[DUWM](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|89.65| 81.35| 2.03| 7.04|MICCAI2020|2020-10|
<!--|Updated [DTC](https://arxiv.org/pdf/2009.04448.pdf)|**89.85**|**81.72**|**1.81**|**7.03**|This repo|2020-10|-->
<!--|Updated DTC (w/o NMS)|89.47|81.09|2.65|9.56|Ours|2020-10|-->
<!--|[SASSNet](https://arxiv.org/pdf/2007.10732.pdf)(w/o NMS)|89.27|80.82|3.13|8.83|MICCAI2020|2020-07|-->
* Note that, the results are very sensitive to the GPU version and PyTorch version, maybe it is good to do experimental just in a fixed GPU and PyTorch.
