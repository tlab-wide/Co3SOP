_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

## change the point cloud range and size to match the dataset
point_cloud_range = [-12.8, -12.8, -2.0, 12.8, 12.8, 2.8]
occ_size = [256, 256, 48]

cam_num = 4
max_connect_car = 0
max_connect_range = 50
use_semantic = True

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

class_names =  ['empty','buildings', 'fences', 'other', 'pedestrians', 'poles',
                'roadlines', 'roads', 'sidewalks', 'vegetation', 'vehicles',
                'walls', 'trafficsigns', 'sky', 'ground','bridge','railtrack', 
                'guardrail', 'trafficlight', 'static', 'dydamic', 'water', 'terrain', 'unlabeled']

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = [192]
_ffn_dim_ = [384]
volume_h_ = [64]
volume_w_ = [64]
volume_z_ = [4]
_num_points_ = [4]
_num_layers_ = [3]

model = dict(
    type='Co3SOPBase',
    cam_num=cam_num,
    car_num=max_connect_car+1,
    use_grid_mask=True,
    use_semantic=use_semantic,
    img_backbone=dict(
       type='ResNet',
       depth=101,
       num_stages=4,
       out_indices=(1,2,3),
       frozen_stages=1,
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,
       style='caffe',
       #with_cp=True, # using checkpoint to save GPU memory
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False), # original DCNv2 will print log when perform load_state_dict
       stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[512, 1024, 2048],
        out_channels=192,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
        ),
    pts_bbox_head=dict(
        type='V2VOccHead',
        max_connect_car=max_connect_car,
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=24,
        conv_input=[192, 96, 96, 48, 48],
        conv_output=[96, 96, 48, 48, 24],
        out_indices=[0,2,4],
        upsample_strides=[1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[128, 128, 128],
        use_semantic=use_semantic,
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            encoder=dict(
                type='Encoder',
                cam_num=cam_num,
                car_num=max_connect_car+1,
                num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=4),
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
        v2v_transformer=dict(
            type="V2VFusionTransformer",
            positional_encoding=dict(
                type='SinePositionalEncoding3D',
                num_feats=_dim_[0] // 3,
                normalize=True),
            embed_dims=_dim_[0],
            num_cars=max_connect_car+1,
            volume_h=volume_h_[0],
            volume_w=volume_w_[0],
            volume_z=volume_z_[0],
            decoder=dict(
            type='DetrTransformerEncoder',
            num_layers=1,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention3D',
                    embed_dims=_dim_[0],
                    num_heads=8,
                    num_levels=max_connect_car,
                    num_points=8,
                    im2col_step=64,
                    dropout=0.0,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None),
                ffn_cfgs=dict(
                    embed_dims=_dim_[0],
                    feedforward_channels=_dim_[0] * 4,
                    ffn_dropout=0.0,),
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')),
            init_cfg=None),
                
        )
),
)

dataset_type = 'Co3SOP'
data_root = 'OPV2V'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img', 'gt_occ'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadOccupancy', use_semantic=use_semantic),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img','gt_occ'])
]

find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train',
        max_connect_car=max_connect_car,
        connect_range=max_connect_range,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR',
        additional_root="additional_25m"),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='validate',
        max_connect_car=max_connect_car,
        connect_range=max_connect_range,
        pipeline=test_pipeline,  
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        modality=input_modality,
        additional_root="additional_25m"),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test',
        max_connect_car=max_connect_car,
        connect_range=max_connect_range,
        pipeline=test_pipeline, 
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        modality=input_modality,
        additional_root="additional_25m"),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
    )

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
