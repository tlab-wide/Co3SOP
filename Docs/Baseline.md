- Train
1. Download [pretrained weights](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth) and put it under Baseline/ckpts/
2. Run training
```python
cd Baseline

CUDA_VISIBLE_DEVICES=0 base tools/dist_train.sh $CONFIG 1 $SAVE_DIR
```


- Evaluation
1. Run testing
```python
cd Baseline

CUDA_VISIBLE_DEVICES=0 base tools/dist_test.sh $CONFIG $CHECKPOINT 1
```