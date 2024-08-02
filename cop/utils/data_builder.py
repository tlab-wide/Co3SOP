import os
import cv2
import open3d as o3d
import numpy as np
from torch.utils import data

class Ego(data.Dataset):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.data_root = configs["dataset_root"]
        self.frames = []
        for scene in os.listdir(self.data_root):
            scene_path = os.path.join(self.data_root, scene)
            if not os.path.isdir(scene_path):
                continue
            for vehicle in os.listdir(scene_path):
                vehicle_path = os.path.join(scene_path, vehicle)
                if not os.path.isdir(vehicle_path):
                   continue
                for frame in os.listdir(vehicle_path):
                    if str(frame).endswith(".yaml"):
                        frame_num = str(frame).split(".")[0]
                        self.frames.append(os.path.join(vehicle_path, frame_num))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        voxels = np.load(frame + ".npy")
        images = []
        for i in range(4):
            image = cv2.imread(frame + f"_camera{i}.png")
            images.append(np.array(image))
        images = np.array(images)
        points = np.asarray(o3d.io.read_point_cloud(frame + ".pcd").points)
        return (images, points, voxels)     

class V2V(data.Dataset):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

    def __getitem__(self, index):
        pass