- Train
1. Download [pretrained weights](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth) and put it under Baseline/ckpts/
2. Ego Stage Training
```python
cd Baseline

CUDA_VISIBLE_DEVICES=0 base tools/dist_train.sh projects/configs/co3sop_base/co3sop_base_{range}_ego.py 1 $SAVE_DIR
```
2. Collaborative Stage Training

Change the `path_to_ego.pth` to the saving checkpoint of Ego Stage in `projects/configs/co3sop_base/co3sop_base_{range}_fusion.py`.
```python
cd Baseline

CUDA_VISIBLE_DEVICES=0 base tools/dist_train.sh projects/configs/co3sop_base/co3sop_base_{range}_fusion.py 1 $SAVE_DIR
```


- Evaluation
1. Run testing
```python
cd Baseline

CUDA_VISIBLE_DEVICES=0 base tools/dist_test.sh $CONFIG $CHECKPOINT 1
```