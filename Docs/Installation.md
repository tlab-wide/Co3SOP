
## Installation for Baseline Model
```python
## conda env
conda create -n co3sop python=3.7 -y
conda activate co3sop
## torch and cuda
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
## mmcv
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
## mmdetection3d
cd Baseline/mmdetection3d
pip install -r requirements/runtime.txt
python setup.py install
cd ../..
##
cd Baseline/extensions/chamfer_dist
python setup.py install --user
cd ../../..
## others
pip install -r Docs/requirements.txt
```

## Installation for Customized Annotation Collection (Optional).

Follow the [Carla official guidelines](https://carla.readthedocs.io/en/0.9.12/start_quickstart/).
