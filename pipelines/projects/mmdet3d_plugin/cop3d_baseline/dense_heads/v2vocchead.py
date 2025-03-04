# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import kornia
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import random
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.cop3d_baseline.loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
import os
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
from mmdet3d.models import builder

def get_discretized_transformation_matrix_3d(matrix, discrete_ratio,
                                          downsample_rate):
    """
    Get disretized transformation matrix.
    Parameters
    ----------
    matrix : torch.Tensor
        Shape -- (B, L, 4, 4) where B is the batch size, L is the max cav
        number.
    discrete_ratio : float
        Discrete ratio.
    downsample_rate : float/int
        downsample_rate

    Returns
    -------
    matrix : torch.Tensor
        Output transformation matrix in 2D with shape (B, L, 3, 4),
        including 2D transformation and 2D rotation.

    """
    matrix = matrix[:, :, [0, 1, 2], :][:, :, :, [0, 1, 2, 3]]
    # normalize the x,y transformation
    matrix[:, :, :, -1] = matrix[:, :, :, -1] \
                          / (discrete_ratio * downsample_rate)

    return matrix.type(dtype=torch.float)

def get_transformation_matrix_3d(M, dsize):
    r"""
    Return transformation matrix for torch.affine_grid.
    Args:
        M : torch.Tensor
            Transformation matrix with shape :math:`(N, 2, 3)`.
        dsize : Tuple[int, int]
            Size of the source image (height, width).

    Returns:
        T : torch.Tensor
            Transformation matrix with shape :math:`(N, 2, 3)`.
    """
    T = get_rotation_matrix3d(M, dsize)
    T[..., 3] += M[..., 3]
    return T

def get_rotation_matrix3d(M, dsize):
    r"""
    Return rotation matrix for torch.affine_grid based on transformation matrix.
    Args:
        M : torch.Tensor
            Transformation matrix with shape :math:`(B, 2, 3)`.
        dsize : Tuple[int, int]
            Size of the source image (height, width).

    Returns:
        R : torch.Tensor
            Rotation matrix with shape :math:`(B, 2, 3)`.
    """
    H, W, Z = dsize
    B = M.shape[0]
    center = torch.Tensor([W / 2, H / 2, Z/12*5]).to(M.dtype).to(M.device).unsqueeze(0)
    # print(center)
    shift_m = eye_like(4, B, M.device, M.dtype)
    shift_m[:, :3, 3] = center

    shift_m_inv = eye_like(4, B, M.device, M.dtype)
    shift_m_inv[:, :3, 3] = -center

    rotat_m = eye_like(4, B, M.device, M.dtype)
    rotat_m[:, :3, :3] = M[:, :3, :3]
    affine_m = shift_m @ rotat_m @ shift_m_inv
    return affine_m[:, :3, :]  # Bx2x3

def eye_like(n, B, device, dtype):
    r"""
    Return a 2-D tensor with ones on the diagonal and
    zeros elsewhere with the same batch size as the input.
    Args:
        n : int
            The number of rows :math:`(n)`.
        B : int
            Btach size.
        device : torch.device
            Devices of the output tensor.
        dtype : torch.dtype
            Data type of the output tensor.

    Returns:
       The identity matrix with the shape :math:`(B, n, n)`.
    """

    identity = torch.eye(n, device=device, dtype=dtype)
    return identity[None].repeat(B, 1, 1)
    
@HEADS.register_module()
class V2VOccHead(nn.Module): 
    def __init__(self,
                 *args,
                 transformer_template=None,
                 v2v_transformer=None,
                 num_classes=24,
                 volume_h=256,
                 volume_w=256,
                 volume_z=64,
                 upsample_strides=[1, 2, 1, 2],
                 out_indices=[0, 2, 4, 6],
                 conv_input=None,
                 conv_output=None,
                 embed_dims=None,
                 img_channels=None,
                 use_semantic=True,
                 freeze=False,
                 max_connect_car=0,
                 **kwargs):
        super(V2VOccHead, self).__init__()
        self.conv_input = conv_input
        self.conv_output = conv_output
        self.freeze = freeze
        self.max_connect_car = max_connect_car
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.img_channels = img_channels
        self.use_semantic = use_semantic
        self.embed_dims = embed_dims
        self.fpn_level = len(self.embed_dims)
        self.upsample_strides = upsample_strides
        self.out_indices = out_indices
        self.transformer_template = transformer_template
        self.v2v_transformer = v2v_transformer
        self._init_layers()
        if self.freeze:
            self._freeze_stages()

    def _init_layers(self):
        self.transformer = nn.ModuleList()
        for i in range(self.fpn_level):
            transformer = copy.deepcopy(self.transformer_template)

            transformer.embed_dims = transformer.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points[i]

            transformer.encoder.transformerlayers.feedforward_channels = \
                self.transformer_template.encoder.transformerlayers.feedforward_channels[i]
            
            transformer.encoder.transformerlayers.embed_dims = \
                self.transformer_template.encoder.transformerlayers.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]
            
            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims[i]
            
            transformer.encoder.num_layers = self.transformer_template.encoder.num_layers[i]

            transformer_i = build_transformer(transformer)
            self.transformer.append(transformer_i)

        if self.v2v_transformer != None and self.max_connect_car > 0:
            self.v2v_fuse = build_transformer(self.v2v_transformer)

        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides
        out_channels = self.conv_output
        in_channels = self.conv_input

        norm_cfg=dict(type='GN', num_groups=24, requires_grad=True)
        upsample_cfg=dict(type='deconv3d', bias=False)
        conv_cfg=dict(type='Conv3d', bias=False)            

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)
            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            self.deblocks.append(deblock)

        self.occ = nn.ModuleList()
        for i in self.out_indices:
            if self.use_semantic:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)

        self.volume_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.volume_embedding.append(nn.Embedding(
                    self.volume_h[i] * self.volume_w[i] * self.volume_z[i], self.embed_dims[i]))

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for i in range(self.fpn_level):
            self.transformer[i].init_weights()
                
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @auto_fp16(apply_to=('mcar_feats'))
    def forward(self, mcar_feats, img_metas):
        car_num = len(mcar_feats)
        volume_embed = []
        for car in range(car_num):
            mlvl_feats = mcar_feats[car]
            bs, num_cam, _, _, _ = mlvl_feats[0].shape
            dtype = mlvl_feats[0].dtype

            volume_queries = self.volume_embedding[0].weight.to(dtype)
            volume_h = self.volume_h[0]
            volume_w = self.volume_w[0]
            volume_z = self.volume_z[0]
            cur_img_metas = [{
                "lidar2img": img_metas[0]["lidar2img"][int(car*num_cam):int(car+1)*num_cam],
                "img_shape": img_metas[0]["img_shape"][int(car*num_cam):int(car+1)*num_cam]
            }]
            volume_embed_i = self.transformer[0](
                    mlvl_feats,
                    volume_queries,
                    volume_h=volume_h,
                    volume_w=volume_w,
                    volume_z=volume_z,
                    img_metas=cur_img_metas
                ).reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)

            volume_embed.append(volume_embed_i)
        
        features = volume_embed

        bs, C, H, W, Z = volume_embed[0].shape
        
        voxel_size = 0.1 * 48 / Z
        batch_transform_matrix = []
        for meta in img_metas:
            transform_matrix = []
            for car in range(car_num):
                matrix = meta["trans2ego"][car]
                S = np.array([[1, 0, 0],[0, 1, 0], [0, 0, -1]])
                matrix[:3,:3] = S @ matrix[:3,:3] @ S
                matrix[:3,3] = S @ matrix[:3,3] 
                transform_matrix.append(matrix)
            batch_transform_matrix.append(transform_matrix)
        batch_transform_matrix = torch.Tensor(np.array(batch_transform_matrix)).to(features[0].device)
        batch_transform_matrix = get_discretized_transformation_matrix_3d(batch_transform_matrix, voxel_size, 1)
        batch_transform_matrix = batch_transform_matrix.view(bs*car_num, 3, 4)
        batch_transform_matrix = get_transformation_matrix_3d(batch_transform_matrix, (H,W,Z)).view(bs, car_num, 3, 4)
        batch_fuse_features = []
        
        for b in range(bs):
            ego_feature = features[0][b,...].unsqueeze(0)
            fuse_features = []
            for neighbor in range(1, car_num):
                neighbor_feature = features[neighbor][b,...].unsqueeze(0)
                if img_metas[0]["vehicle_id"][0] == img_metas[0]["vehicle_id"][neighbor]:
                    neighbor_feature = neighbor_feature*0
                else:
                    neighbor_feature = kornia.geometry.transform.warp_affine3d(
                                neighbor_feature.permute(0,1,4,3,2), 
                                batch_transform_matrix[b,neighbor,:3,:].unsqueeze(0), (Z,W,H), 
                                flags='bilinear',
                                padding_mode='zeros', 
                                align_corners=True)
                    neighbor_feature = neighbor_feature.permute(0,1,4,3,2)
                fuse_features.append(neighbor_feature)

            if len(fuse_features) > 0:
                fuse_features = self.v2v_fuse(
                    fuse_features,
                    ego_feature,
                    H,W,Z,
                    img_metas=img_metas).reshape(
                        bs, H, W, Z, -1).permute(0, 4, 1, 2, 3)
            else:
                fuse_features = ego_feature
            batch_fuse_features.append(fuse_features)

        batch_fuse_features = torch.cat(batch_fuse_features, dim=0)
        
        outputs = []
        result = batch_fuse_features
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)
            if i in self.out_indices:
                outputs.append(result)

        occ_preds = []
        for i in range(len(outputs)):
            occ_pred = self.occ[i](outputs[i])
            occ_preds.append(occ_pred)

        outs = {
            'volume_embed': volume_embed,
            'occ_preds': occ_preds,
        }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_occ,
             preds_dicts,
             img_metas):

        if not self.use_semantic:
            loss_dict = {}
            for i in range(len(preds_dicts['occ_preds'])):

                pred = preds_dicts['occ_preds'][i][:, 0]

                ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)

                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                    
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))
                    
                loss_occ_i =  loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i)) #* focal_weight

                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i
    
        else:
            pred = preds_dicts['occ_preds']
            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )
            loss_dict = {}
            for i in range(len(preds_dicts['occ_preds'])):
                pred = preds_dicts['occ_preds'][i].float()
                ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)
                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                criterion_loss = criterion(pred, gt.long())
                sem_loss = sem_scal_loss(pred, gt.long())
                gep_loss = geo_scal_loss(pred, gt.long())
                loss_occ_i = (criterion_loss+gep_loss+sem_loss)
                loss_occ_i = loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))
                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        return loss_dict

    def _freeze_stages(self):
        for layer in self.transformer:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.deblocks:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.volume_embedding:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.transfer_conv:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

        for layer in self.occ:
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode 
        while keep normalization layerfreezed."""
        super(V2VOccHead, self).train(mode)
        if self.freeze:
            self._freeze_stages()
        
