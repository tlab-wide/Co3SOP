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
        additional/
            train/
            validate/
            test/
```