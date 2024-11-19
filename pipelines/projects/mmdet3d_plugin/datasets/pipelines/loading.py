#import open3d as o3d
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import random
import os


@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load occupancy groundtruth.

    Expects results['occ_path'] to be a list of filenames.

    The ground truth is a (N, 4) tensor, N is the occupied voxel number,
    The first three channels represent xyz voxel coordinate and last channel is semantic class. 
    """

    def __init__(self, use_semantic=True):
        self.use_semantic = use_semantic

    
    def __call__(self, results):
        occ_size = results["occ_size"]
        voxels = np.load(results['occ_path'])["voxels"]
        f = int((voxels.shape[0]-occ_size[0])/2)
        b = int((voxels.shape[0]+occ_size[0])/2)
        l = int((voxels.shape[1]-occ_size[1])/2)
        r = int((voxels.shape[1]+occ_size[1])/2)
        voxels = voxels[f:b,l:r,:occ_size[2]]
        voxels[voxels==0] = 23
        voxels[voxels==255] = 0
        
        results['gt_occ'] = voxels
        
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

