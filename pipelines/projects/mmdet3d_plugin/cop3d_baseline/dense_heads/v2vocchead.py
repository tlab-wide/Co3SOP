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
from projects.mmdet3d_plugin.cop3d_baseline.modules.convgru3d import ConvGRU3D
import os
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse

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

@HEADS.register_module()
class V2VOccHead(nn.Module): 
    def __init__(self,
                 *args,
                 transformer_template=None,
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



        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg=dict(type='deconv3d', bias=False)
        conv_cfg=dict(type='Conv3d', bias=False)

        self.v2v_fuse = nn.Sequential(
            build_conv_layer(conv_cfg, 
                             in_channels=out_channels[6]*(self.max_connect_car+1), 
                             out_channels=out_channels[6]*(self.max_connect_car+1), 
                             kernel_size=3, 
                             stride=1, 
                             padding=1),
            build_norm_layer(norm_cfg, out_channels[6]*(self.max_connect_car+1))[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, 
                             in_channels=out_channels[6]*(self.max_connect_car+1), 
                             out_channels=out_channels[6], 
                             kernel_size=3, 
                             stride=1, 
                             padding=1),
            build_norm_layer(norm_cfg, out_channels[6])[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, 
                             in_channels=out_channels[6], 
                             out_channels=out_channels[6], 
                             kernel_size=3, 
                             stride=1, 
                             padding=1),
            build_norm_layer(norm_cfg, out_channels[6])[1],
            nn.ReLU(inplace=True))
        
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


        self.transfer_conv = nn.ModuleList()
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        conv_cfg=dict(type='Conv2d', bias=True)
        for i in range(self.fpn_level):
            transfer_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=self.img_channels[i],
                    out_channels=self.embed_dims[i],
                    kernel_size=1,
                    stride=1)
            transfer_block = nn.Sequential(transfer_layer,
                    nn.ReLU(inplace=True))

            self.transfer_conv.append(transfer_block)
        

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for i in range(self.fpn_level):
            self.transformer[i].init_weights()
                
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):
        # print(img_metas)
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        car_num = num_cam//4

        volume_embed = []
        for i in range(self.fpn_level):
            volume_queries = self.volume_embedding[i].weight.to(dtype)
            
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]
            _, _, C, H, W = mlvl_feats[i].shape
            volume_embed_i = []
            for j in range(car_num):
                car_img_meta = [{
                    "lidar2img": img_metas[0]["lidar2img"][j*4:(j+1)*4],
                    "img_shape": img_metas[0]["img_shape"]
                }]
                view_features = self.transfer_conv[i](mlvl_feats[i][:,j*4:(j+1)*4,...].reshape(bs*4, C, H, W)).reshape(bs, 4, -1, H, W)
            # print(view_features.shape)
                volume_embed_i_j = self.transformer[i](
                    [view_features],
                    volume_queries,
                    volume_h=volume_h,
                    volume_w=volume_w,
                    volume_z=volume_z,
                    img_metas=car_img_meta
                )
                volume_embed_i.append(volume_embed_i_j)
            volume_embed.append(torch.cat(volume_embed_i, dim=0))
        

        volume_embed_reshape = []
        for i in range(self.fpn_level):
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            volume_embed_reshape_i = volume_embed[i].reshape(bs*car_num, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)
            # print(volume_embed_reshape_i.shape)
            volume_embed_reshape.append(volume_embed_reshape_i)
            # volume_embed_reshape.append(volume_embed_reshape_i)
        
        outputs = []
        # print(len(volume_embed_reshape))
        result = volume_embed_reshape.pop()
        # print(result.shape)
        for i in range(len(self.deblocks)):
            # print(result.shape)
            result = self.deblocks[i](result)
            # print(result.shape)
            if i in self.out_indices:
                outputs.append(result)
                # print(result.shape)
            elif i in [1, 3] and len(volume_embed_reshape)>0:  # we do not add skip connection at level 0
                volume_embed_temp = volume_embed_reshape.pop()
                # print(len(volume_embed_reshape))
                # print(volume_embed_temp.shape)
                result = result + volume_embed_temp

        features = outputs.pop()
        _, C, H, W, Z = features.shape
        features = features.reshape(bs, car_num, C, H, W, Z)
        
        voxel_size = 0.1 * 48 / Z

        # batch_fuse_features = []
        batch_transform_matrix = []
        for meta in img_metas:
            transform_matrix = []
            for car in range(car_num):
                transform_matrix.append(meta["trans2ego"][car])
            # transform_matrix = torch.Tensor(transform_matrix)
            batch_transform_matrix.append(transform_matrix)
        batch_transform_matrix = torch.Tensor(np.array(batch_transform_matrix)).cuda()
        batch_transform_matrix = get_discretized_transformation_matrix_3d(batch_transform_matrix, voxel_size, 1)

        
        # print("before transformation",torch.max(ego_feature))
        # print(batch_transform_matrix[:,0,:3,:])
        # ego_feature = kornia.geometry.transform.warp_affine3d(
        #                     ego_feature.permute(0,1,4,2,3), 
        #                     batch_transform_matrix[:,0,:3,:], (Z,H,W), 
        #                     flags='nearest', 
        #                     padding_mode='zeros', 
        #                     align_corners=True).permute(0,1,3,4,2)
        # print("after transformation",torch.max(ego_feature))

        # if car_num > 1:
            
            # neighbor_features = features[:,1:,...].reshape(bs*(car_num-1), C, H, W, Z)
            # neighbor_feature = kornia.geometry.transform.warp_affine3d(
            #             neighbor_features.permute(0,1,4,2,3), 
            #             batch_transform_matrix[:,:,:3,:].reshape(bs*(car_num-1), 3, 4), (Z,H,W), 
            #             flags='bilinear', 
            #             padding_mode='zeros', 
            #             align_corners=True).permute(0,1,3,4,2)
            # fuse_features = torch.cat([neighbor_features, ego_feature.repeat(car_num-1,1,1,1,1)], dim=1)
            # fuse_features = self.v2v_fuse_conv_stage1(fuse_features).reshape(bs, car_num-1, -1, H, W, Z)
            # if len(fuse_features) > 0:
            #     ego_feature = self.v2v_fuse_conv_stage2(torch.cat([ego_feature, torch.mean(fuse_features, dim=1)], dim=1))
        batch_fuse_features = []
        for b in range(bs):
            ego_feature = features[b,0,...].unsqueeze(0)
            neighbor_num = 0
            fuse_features = [ego_feature]
            for neighbor in range(1, car_num):
                
                
                neighbor_feature = features[b,neighbor,...].unsqueeze(0)

                # print("before transformation, non 0 num:", (neighbor_feature!=0).sum())
                if img_metas[0]["vehicle_id"][0] == img_metas[0]["vehicle_id"][neighbor]:
                    neighbor_feature[neighbor_feature!=0] = 0
                else:
                    neighbor_feature = kornia.geometry.transform.warp_affine3d(
                                neighbor_feature.permute(0,1,4,2,3), 
                                batch_transform_matrix[:,neighbor,:3,:], (Z,H,W), 
                                flags='nearest', 
                                padding_mode='zeros', 
                                align_corners=True)
                    neighbor_feature = neighbor_feature.permute(0,1,3,4,2)
                    # ego_feature = torch.where(ego_feature == 0, neighbor_feature, ego_feature)
                    # print(((neighbor_feature!=0)*(ego_feature!=0)).sum())
                # print(neighbor_feature.shape)
                # if (neighbor_feature!=0).sum() == 0:
                #     continue
                neighbor_num = neighbor_num + 1
                # print("after transformation, non 0 num:", (neighbor_feature!=0).sum())
                # fuse_feature = torch.cat([neighbor_feature, ego_feature], dim=1)
                # ego_feature = self.v2v_fuse(fuse_feature)
                # # print(fuse_feature.shape)
                # mask = (neighbor_feature!=0).float()
                # fuse_feature = self.v2v_fuse_conv_stage1(fuse_feature) * mask
                
                # neighbor_feature = neighbor_feature*mask
                fuse_features.append(neighbor_feature)
            # fuse_features = ego_feature
            # random.shuffle(fuse_features)
            fuse_features = torch.cat(fuse_features, dim=1)
            fuse_features =  self.v2v_fuse(fuse_features)
            # # print(fuse_features.shape)
            # fuse_features = self.mlp(fuse_features.permute(0,2,3,4,1).contiguous()).permute(0,4,1,2,3).contiguous()
            # print(fuse_features.shape)
            # print(neighbor_num)
            # if len(fuse_features) > 0:
            #     fuse_features = self.v2v_fuse_conv_stage2(
            #         torch.cat(
            #             [ego_feature, 
            #             torch.mean(torch.cat(fuse_features, dim=0), dim=0).unsqueeze(0)], 
            #         dim=1).unsqueeze(1))[0][0].squeeze(0)
            #     # ego_feature = self.v2v_fuse_conv_stage2(
            #     #     torch.cat(
            #     #         [ego_feature, 
            #     #         torch.mean(torch.cat(fuse_features, dim=0), dim=0).unsqueeze(0)], 
            #     #         dim=1))
            # # print(type(ego_feature[1]))
                # ego_feature = self.mlp(fuse_features.permute(0,2,3,4,1).contiguous()).permute(0,4,1,2,3).contiguous()
            batch_fuse_features.append(fuse_features)
        batch_fuse_features = torch.cat(batch_fuse_features, dim=0)
        outputs.append(batch_fuse_features)



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
                # print(preds_dicts['occ_preds'][i].shape)
                ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)

                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                
                #gt = torch.mode(gt, dim=-1)[0].float()
                    
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))
                    
                loss_occ_i =  loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i)) #* focal_weight

                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i
    
        else:
            pred = preds_dicts['occ_preds']
            
            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )
            
            loss_dict = {}
            # print(len(preds_dicts['occ_preds']))
            # for i in range(len(preds_dicts['occ_preds'])):
            #     print(preds_dicts['occ_preds'][i].shape)
            for i in range(len(preds_dicts['occ_preds'])):
                
                pred = preds_dicts['occ_preds'][i].float()
                # print(pred.shape)
                # if pred.shape[4] != 24:
                #     continue
                # ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)
                # print(preds_dicts['occ_preds'][i].shape)
                gt = gt_occ.long() # multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)
                criterion_loss = criterion(pred, gt)
                sem_loss = sem_scal_loss(pred, gt)
                gep_loss = geo_scal_loss(pred, gt)
                # print(f"criterion:{criterion_loss}")
                # print(f"sem_loss:{sem_loss}")
                # print(f"geo_loss:{gep_loss}")

                loss_occ_i = (criterion_loss+gep_loss+sem_loss)

                # loss_occ_i = loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))

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

        # for layer in self.occ:
        #     layer.eval()
        #     for param in layer.parameters():
        #         param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(V2VOccHead, self).train(mode)
        if self.freeze:
            print("freeze fpn")
            self._freeze_stages()
        
