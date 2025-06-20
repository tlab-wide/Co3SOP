
from termios import BS0
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.utils import build_transformer
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .spatial_cross_attention import MSDeformableAttention3D
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.transformer import build_positional_encoding
from projects.mmdet3d_plugin.co3sop_base.modules.point_generator import MlvlPointGenerator

@TRANSFORMER.register_module()
class V2VFusionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self, volume_h, volume_w, volume_z,
                 num_feature_levels=4,
                 num_cars=4,
                 two_stage_num_proposals=300,
                 decoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 rotate_center=[256, 256],
                 positional_encoding=None,
                 **kwargs):
        super(V2VFusionTransformer, self).__init__(**kwargs)

        self.decoder = build_transformer_layer_sequence(decoder)
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(positional_encoding)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cars = num_cars
        self.fp16_enabled = False
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z

        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center
        self.point_generator = MlvlPointGenerator([1])


    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Embedding(self.num_cars, self.embed_dims)
        self.cars_embeds = nn.Parameter(
            torch.Tensor(self.num_cars, self.embed_dims))
        self.volume_query = nn.Embedding(
                    self.volume_h * self.volume_w * self.volume_z, self.embed_dims)
        
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cars_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    @staticmethod
    def get_reference_points(H, W, Z, bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in SCA and TSA.
        Args:
            H, W, Z: spatial shape of volume.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,
                            device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                            device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                            device=device).view(1, H, 1).expand(Z, H, W) / H
        

        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2)
        ref_3d = ref_3d.flatten(1)

        ref_3d = ref_3d.permute(1, 0)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1)
        return ref_3d
    
    @auto_fp16(apply_to=('voxel_feats', 'volume_queries'))
    def forward(
            self,
            voxel_feats,
            volume_queries,
            volume_z,
            volume_h,
            volume_w,
            **kwargs):

        bs = voxel_feats[0].size(0)
        bs, c, z, y, x = volume_queries.shape
        padding_mask_resized = voxel_feats[0].new_zeros((bs, ) + volume_queries.shape[-3:], dtype=torch.bool)
        query_pos = self.positional_encoding(padding_mask_resized)
        reference_points = self.get_reference_points(x, y, z, bs=bs, device=volume_queries.device)
        # (h_i * w_i * d_i, 2)
        # reference_points = self.point_generator.single_level_grid_priors(
        #     volume_queries.shape[-3:], 0, device=volume_queries.device)
        # normalize points to [0, 1]
        # factor = volume_queries.new_tensor([[x, y, z]])
        # reference_points = reference_points / factor
        volume_queries = volume_queries.flatten(2).permute(0,2,1)
        query_pos = query_pos.flatten(2).permute(0,2,1)
        feat_flatten = []
        spatial_shapes = []
        key_pos_list=[]
        voxel_masks = []
        for lvl, feat in enumerate(voxel_feats):
            padding_mask_resized = feat.new_zeros((bs, ) + feat.shape[-3:], dtype=torch.bool)
            # key_pos = self.positional_encoding(padding_mask_resized).flatten(2).permute(2,0,1)
            _, _, z_i, y_i, x_i = feat.shape
            # spatial_shape = (z_i, y_i, x_i)
            # key_pos_list.append(key_pos)
            
            feat = feat.flatten(2).permute(2,0,1) # bs, hwz, c
            voxel_mask = (feat.abs().sum(-1)>0).unsqueeze(0)
            # print(voxel_mask.shape)
            feat_flatten.append(feat)
            voxel_masks.append(voxel_mask)
        spatial_shapes.append((z, y, x))
        feat_flatten = torch.stack(feat_flatten, 0)
        voxel_masks = torch.stack(voxel_masks, 0)
        # print(feat_flatten.shape)
        # print(volume_queries.shape)
        # print(voxel_masks.shape)
        
        # key_pos_list = torch.cat(key_pos_list, dim=0)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)

        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = reference_points.permute(1, 0, 2, 3).view(
            1, bs, 1, x*y*z, 3).repeat(1, 1, len(voxel_feats), 1, 1).permute(2, 1, 3, 0, 4)
        # print(reference_points.shape)
        volume_embed = self.decoder(
                query=volume_queries,
                key=feat_flatten,
                value=feat_flatten,
                query_pos=query_pos,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                query_key_padding_mask=None,
                spatial_shapes=spatial_shapes,
                reference_points_car=reference_points,
                level_start_index=level_start_index,
                bev_mask=voxel_masks,
                **kwargs
            )

        return volume_embed


