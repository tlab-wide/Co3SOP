1. Download the [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) dataset.

2. Download our annotation for 3D semantic occupancy prediction from [here](https://huggingface.co/datasets/WuHanlin1997/Co3SOP/tree/main).

3. The dataset folder should be structured as follows:
```python
Baseline/
    OPV2V/
        ## OPV2V 
        train/
        test/
        validate/
        ## Additional Annotation
        additional_25m/
            train/
            validate/
            test/
        additional_50m/
            train/
            validate/
            test/
        additional_76m/
            train/
            validate/
            test/
```

4. Notes: Each annotation file is named {frame_idx}_voxels.npz. Each file contains `[voxels]`, for example:
```python
import numpy as np
filename = "OPV2V/additional_25m/train/2021_08_16_22_26_54/641/000069_voxels.npz"
voxels = np.load(filename)["voxels"]
```