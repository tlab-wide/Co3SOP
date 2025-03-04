# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

#import open3d as o3d
from tkinter.messagebox import NO
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from projects.mmdet3d_plugin.datasets.evaluation_metrics import evaluation_reconstruction, evaluation_semantic
from sklearn.metrics import confusion_matrix as CM
import time, yaml, os
import torch.nn as nn
import pdb


@DETECTORS.register_module()
class Co3SOPBase(MVXTwoStageDetector):
    def __init__(self,
                 cam_num=4,
                 car_num=4,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_semantic=True,
                 is_vis=False,
                 version='v1',
                 ):

        super(Co3SOPBase,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        self.cam_num = cam_num
        self.car_num = car_num

        self.use_semantic = use_semantic
        self.is_vis = is_vis

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        
        if img is not None:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        # print("1")
        # print(img.shape)
        B, N, C, H, W = img.shape
        assert N == self.cam_num*self.car_num
        img = img.reshape(B, self.car_num, self.cam_num, C, H, W)
        multi_img_feats=[]
        for i in range(self.car_num):
            img_i = img[:,i,:,:,:,:].contiguous()
            img_feats = self.extract_img_feat(img_i, img_metas, len_queue=len_queue)
            multi_img_feats.append(img_feats)
        return multi_img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_occ,
                          img_metas):
        outs = self.pts_bbox_head(
            pts_feats, img_metas)

        loss_inputs = [gt_occ, outs]

        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # print("1")
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      gt_occ=None,
                      img=None
                      ):

        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_occ,
                                             img_metas)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, gt_occ=None, **kwargs):
        # print("1")
        output = self.simple_test(
            img_metas, img, **kwargs)
        
        pred_occ = output['occ_preds']
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]

        if self.use_semantic:
            class_num = pred_occ.shape[1]
            _, pred_occ = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
            eval_results = evaluation_semantic(pred_occ, gt_occ, img_metas[0], class_num)
        else:
            pred_occ = torch.sigmoid(pred_occ[:, 0])
            eval_results = evaluation_reconstruction(pred_occ, gt_occ, img_metas[0])
        return {'evaluation': eval_results}
        

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas)

        return outs

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        # print("1")
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        output = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)

        return output
    
    
    