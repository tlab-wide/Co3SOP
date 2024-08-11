import os
import cv2
import numpy as np
from torch.utils import data

class Ego(data.Dataset):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        self.root = configs["dataset"]["root"]
        self.additional = configs["dataset"]["additional"]
        self.dimensions = configs["model_params"]["input_dimensions"]
        self.frames = []
        for scene in os.listdir(self.root):
            scene_path = os.path.join(self.root, scene)
            if scene == "2021_09_09_13_20_58":
                continue
            if not os.path.isdir(scene_path):
                continue
            for vehicle in os.listdir(scene_path):
                vehicle_path = os.path.join(scene_path, vehicle)
                if not os.path.isdir(vehicle_path):
                   continue
                for frame in os.listdir(vehicle_path):
                    if str(frame).endswith(".pcd"):
                        frame_num = str(frame).split(".")[0]
                        self.frames.append(os.path.join(vehicle_path, frame_num))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        images = []
        # print(frame)
        # for i in range(4):
        #     image = cv2.imread(frame + f"_camera{i}.png")
        #     images.append(np.array(image))
        # images = np.array(images)
        points = [] #np.asarray(o3d.io.read_point_cloud(frame + ".pcd").points)
        occupancy = np.load(str(frame).replace(self.root, self.additional) + "_occupancy.npz")["voxels"]
        voxels = []
        voxels = np.load(str(frame).replace(self.root, self.additional) + "_voxels.npz")["voxels"]
        f = int((occupancy.shape[0]-self.dimensions[0])/2)
        b = int((occupancy.shape[0]+self.dimensions[0])/2)
        l = int((occupancy.shape[1]-self.dimensions[1])/2)
        r = int((occupancy.shape[1]+self.dimensions[1])/2)

        occupancy = occupancy[f:b,l:r,:self.dimensions[2]]
        voxels = voxels[f:b,l:r,:self.dimensions[2]]
        voxels[voxels==255] = 23
        # print(occupancy.shape)
        ## "images": images.astype(np.float32), 
        return {"occupancy": occupancy.astype(np.float32), "voxels": voxels.astype(np.float32)}  

class V2V(data.Dataset):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs

    def __getitem__(self, index):
        pass