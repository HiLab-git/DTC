# import torch
# from torch.nn import functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

def compute_dtm(img_gt, out_shape, normalize=False, fg=False):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if not fg:
            if posmask.any():
                negmask = 1 - posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    fg_dtm[b] = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) + (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                else:
                    fg_dtm[b] = posdis + negdis
                fg_dtm[b][boundary==1] = 0
        else:
            if posmask.any():
                posdis = distance(posmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                if normalize:
                    fg_dtm[b] = (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                else:
                    fg_dtm[b] = posdis
                fg_dtm[b][boundary==1] = 0

    return fg_dtm

def hd_loss(seg_soft, gt, gt_dtm=None, one_side=True, seg_dtm=None):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft - gt.float()) ** 2
    g_dtm = gt_dtm ** 2
    dtm = g_dtm if one_side else g_dtm + seg_dtm ** 2
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    # hd_loss = multipled.sum()*1.0/(gt_dtm > 0).sum()
    hd_loss = multipled.mean()

    return hd_loss



def save_sdf(gt_path=None):
    '''
    generate SDM for gt segmentation
    '''
    import nibabel as nib
    dir_path = 'C:/Seolen/PycharmProjects/semi_seg/semantic-semi-supervised-master/model/gan_sdfloss3D_0229_04/test'
    gt_path = dir_path + '/00_gt.nii.gz'
    gt_img = nib.load(gt_path)
    gt = gt_img.get_data().astype(np.uint8)
    posmask = gt.astype(np.bool)
    negmask = ~posmask
    posdis = distance(posmask)
    negdis = distance(negmask)
    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
    # sdf = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis)) - (posdis - np.min(posdis)) / ( np.max(posdis) - np.min(posdis))
    sdf = (posdis - np.min(posdis)) / ( np.max(posdis) - np.min(posdis))
    sdf[boundary==1] = 0
    sdf = sdf.astype(np.float32)

    sdf = nib.Nifti1Image(sdf, gt_img.affine)
    save_path = dir_path + '/00_sdm_pos.nii.gz'
    nib.save(sdf, save_path)



def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def sdf_loss(net_output, gt_sdm):
    # print('net_output.shape, gt_sdm.shape', net_output.shape, gt_sdm.shape)
    # ([4, 1, 112, 112, 80])

    smooth = 1e-5
    # compute eq (4)
    intersect = torch.sum(net_output * gt_sdm)
    pd_sum = torch.sum(net_output ** 2)
    gt_sum = torch.sum(gt_sdm ** 2)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF = 1/3 - L_product + torch.norm(net_output - gt_sdm, 1)/torch.numel(net_output)

    return L_SDF





def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """
    pc = outputs_soft[:,1,...]
    dc = gt_sdf[:,1,...]
    multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
    bd_loss = multipled.mean()

    return bd_loss

if __name__ == '__main__':
    save_sdf()