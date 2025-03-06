import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import tempfile
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
import pdb, os
import yaml
from projects.mmdet3d_plugin.models.utils.transformation_utils import cal_dist, x1_to_x2


@DATASETS.register_module()
class Co3SOP(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, occ_size, pc_range, use_semantic=False, classes=None, overlap_test=False, 
                 max_connect_car=0, connect_range= 50, additional_root="additional", *args, **kwargs):
        self.occ_size = occ_size
        self.additional_root = additional_root
        super().__init__(*args, **kwargs)
        
        self.overlap_test = overlap_test
        self.max_connect_car = max_connect_car
        self.pc_range = pc_range
        self.use_semantic = use_semantic
        self.class_names = classes
        self._set_group_flag()
        self.connect_range = connect_range
        
        
    def load_annotations(self, ann_file):
        data_infos = []
        self.train_data_root = os.path.join(self.data_root, self.ann_file)
        self.additional_train_data = os.path.join(self.data_root, f"{self.additional_root}/{self.ann_file}")
        self.scenes = []
        self.vehicle_infos = {}
        data_infos = []
        self.frames = []
        scene_num = 0
        for scene in os.listdir(self.train_data_root):
            scene_path = os.path.join(self.train_data_root, scene)
            # if scene == "2021_09_09_13_20_58":
            #     continue
            if not os.path.isdir(scene_path):
                continue
            scene_num = scene_num+1
            self.scenes.append(scene)
            self.vehicle_infos[scene] = {
                "vehicles":[],
                "frames":[]
            }
            i = 0
            for vehicle in os.listdir(scene_path):
                vehicle_path = os.path.join(scene_path, vehicle)
                if not os.path.isdir(vehicle_path):
                   continue
                self.vehicle_infos[scene]["vehicles"].append(vehicle)
                for frame in os.listdir(vehicle_path):
                    splits = str(frame).split(".")
                    if splits[1] == 'yaml' and splits[0].isdigit():
                        self.vehicle_infos[scene]["frames"].append(splits[0])
                        data_infos.append((scene, vehicle, splits[0]))
        return data_infos
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        scene, vehicle, frame_num = self.data_infos[index]

        neighbors = [x for x in self.vehicle_infos[scene]["vehicles"] if x != vehicle]
        frame = os.path.join(self.train_data_root, scene, vehicle, frame_num)
        data = {}
        data["occ_path"] = str(frame).replace(self.train_data_root, self.additional_train_data) + "_voxels.npz"
        data["occ_size"] = self.occ_size
        data["pc_range"] = self.pc_range
        data["img_filename"] = []
        data["lidar2img"] = []
        data["lidar2cams"] = []
        data["cam_intrinsic"] = []
        data["pose"] = []
        data["trans2ego"] = []
        data["vehicle_id"] = []

        ego_info = self.get_vehicle_data(scene, vehicle, frame_num)
        data["img_filename"].extend(ego_info["imgs"])
        data["cam_intrinsic"].extend(ego_info["intrins"])
        data["lidar2img"].extend(ego_info["lidar2img"])
        data["lidar2cams"].extend(ego_info["lidar2cams"])
        data["vehicle_id"].append(ego_info["vehicle_id"])
        data["pose"].append(ego_info["pose"])
        data["trans2ego"].append(np.asarray(x1_to_x2(ego_info["pose"], ego_info["pose"])))
        neighbor_num = 0
        near_neighbor = []
        for neighbor in neighbors:
            neighbor_info = self.get_vehicle_data(scene, neighbor, frame_num)

            if cal_dist(neighbor_info["pose"], ego_info["pose"]) > self.connect_range:
                continue

            near_neighbor.append(neighbor_info)

        near_neighbor = sorted(near_neighbor, key=lambda neighbor: cal_dist(neighbor["pose"], ego_info["pose"]))
        for neighbor_info in near_neighbor:
            if neighbor_num >= self.max_connect_car:
                break
            data["img_filename"].extend(neighbor_info["imgs"])
            data["cam_intrinsic"].extend(neighbor_info["intrins"])
            data["lidar2cams"].extend(neighbor_info["lidar2cams"])
            data["lidar2img"].extend(neighbor_info["lidar2img"])
            data["vehicle_id"].append(neighbor_info["vehicle_id"])
            data["pose"].append(neighbor_info["pose"])
            data["trans2ego"].append(np.asarray(x1_to_x2(neighbor_info["pose"], ego_info["pose"])))
            neighbor_num = neighbor_num +1

        for _ in range(self.max_connect_car-neighbor_num):
            data["img_filename"].extend(ego_info["imgs"])
            data["cam_intrinsic"].extend(ego_info["intrins"])
            data["lidar2cams"].extend(ego_info["lidar2cams"])
            data["lidar2img"].extend(ego_info["lidar2img"])
            data["vehicle_id"].append(ego_info["vehicle_id"])
            data["pose"].append(ego_info["pose"])
            data["trans2ego"].append(np.asarray(x1_to_x2(ego_info["pose"], ego_info["pose"])))
        
        return data

    def get_vehicle_data(self, scene, vehicle, frame_num):
        frame = os.path.join(self.train_data_root, scene, vehicle, frame_num)
        
        meta_data = {}
        with open(frame+'.yaml','r') as f:
            meta_data = yaml.load(f, yaml.UnsafeLoader)
        pose = np.array(meta_data["lidar_pose"])
        imgs = []
        intrins = []
        lidar2cams = []
        lidar2imgs = []
        for i in range(4):
            imgs.append(frame + f"_camera{i}.png")

            intrin = np.array(meta_data[f"camera{i}"]["intrinsic"])
            viewpad = np.eye(4)
            viewpad[:intrin.shape[0], :intrin.shape[1]] = intrin
            ## carla coordinate to opencv coordinate
            axis_trans = np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]])
            lidar2img = np.array(viewpad @ axis_trans @ np.array(meta_data[f"camera{i}"]["extrinsic"]))

            intrins.append(viewpad)
            lidar2cams.append(np.array(meta_data[f"camera{i}"]["extrinsic"]))
            lidar2imgs.append(lidar2img)

        data = {
            "imgs": imgs,
            "vehicle_id": vehicle,
            "intrins": intrins,
            "lidar2img": lidar2imgs,
            "lidar2cams": lidar2cams,
            "pose": pose,
        }
        return data

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            info = self.data_infos[idx]
            
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        return results, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        
        results, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = {}
        if self.use_semantic:
            class_names = {}
            class_num = len(self.class_names)
            for i, name in enumerate(self.class_names):
                class_names[i ] = self.class_names[i]
            
            results = np.stack(results, axis=0).mean(0)
            mean_ious = []
            
            for i in range(class_num):
                tp = results[i, 0]
                p = results[i, 1]
                g = results[i, 2]
                union = p + g - tp+0.00001
                mean_ious.append(tp / union)
            
            for i in range(class_num):
                results_dict[class_names[i]] = mean_ious[i]
            results_dict['mIoU'] = np.mean(np.array(mean_ious)[1:])


        else:
            results = np.stack(results, axis=0).mean(0)
            results_dict={'Acc':results[0],
                          'Comp':results[1],
                          'CD':results[2],
                          'Prec':results[3],
                          'Recall':results[4],
                          'F-score':results[5]}

        return results_dict
