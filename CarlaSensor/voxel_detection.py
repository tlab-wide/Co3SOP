"""
This is mainly used to filter out objects that is not in the sight
of cameras.
"""
import weakref
import os
import time
import carla
import cv2
import numpy as np
from multiprocessing import Process
from mayavi import mlab
from logreplay.sensors.base_sensor import BaseSensor


class VoxelDetection(BaseSensor):
    def __init__(self, agent_id, vehicle, world, config, global_position):
        super().__init__(agent_id, vehicle, world, config, global_position)

        if vehicle is not None:
            self.world = vehicle.get_world()

        self.agent_id = agent_id

        blueprint = self.world.get_blueprint_library(). \
            find('sensor.other.voxel_detection')
        # set attribute based on the configuration
        self.detect_range = int(config["detect_range"])
        self.voxel_size = float(config["voxel_size"])
        self.top = float(config["top"])
        self.bottom = float(config["bottom"])
        self.self_ignore = int(config['self_ignore'])
        blueprint.set_attribute('detected_len', str(self.detect_range))
        blueprint.set_attribute('top_boundary', str(self.top))
        blueprint.set_attribute('bottom_boundary', str(self.bottom))
        blueprint.set_attribute('box_size', str(self.voxel_size))
        blueprint.set_attribute('self_ignore', str(self.self_ignore))
        blueprint.set_attribute('draw_debug', str(int(config['draw_debug'])))
        
        spawn_point = carla.Transform(carla.Location(z=0))

        self.name = 'voxel_detection' + str(self.agent_id)

        if vehicle is not None:
            self.sensor = self.world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = self.world.spawn_actor(blueprint, spawn_point)

        self.x_len = int(2*self.detect_range/self.voxel_size)
        self.y_len = int(2*self.detect_range/self.voxel_size)
        self.z_len = int((self.top-self.bottom)/self.voxel_size)
        self.voxels = np.zeros((self.x_len, self.y_len, self.z_len),dtype=np.int8)

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: VoxelDetection._on_data_event(
                weak_self, event))
        
        self.draw_process = None
        self.data = None
        self.Update = False

    @staticmethod
    def _on_data_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.frame = event.frame
        self.timestamp = event.timestamp
        self.data = event
        self.Update = True
        print("set voxel")
        

    def tick(self):
        
        return None
    
    def data_dump(self, output_root, cur_timestamp):
        while not self.Update:
            time.sleep(1)
        timestamp = cur_timestamp.split("\\")[-1]
        time.sleep(1)
        voxel_array = []
        for data in self.data:
            voxel_array.append(data)
        voxel_array = np.array(voxel_array, dtype=np.uint8)

        self.voxels = voxel_array.reshape((self.x_len, self.y_len, self.z_len))
        output_vis_name = os.path.join(
            output_root, 
            timestamp+f'_voxels.npz')
        print(output_vis_name)
        np.savez_compressed(output_vis_name, voxels=self.voxels)
        print("save voxel")
        self.Update = False

    @staticmethod
    def draw(voxels:np.array,id):
        figure = mlab.figure(figure=f"{id}",size=(2560, 1440), bgcolor=(1, 1, 1))
        x = []
        y = []
        z = []
        s = []
        shape = voxels.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if voxels[x][y][z] >= 0:
                        x.append(i)
                        y.append(j)
                        z.append(k)
                        s.append(voxels[x][y][z])
        plt_plot_fov = mlab.points3d(
            x,
            y,
            z,
            s,
            colormap="viridis",
            scale_factor = 0.2,
            mode="cube",
            opacity=1.0,
            vmin=1,
            vmax=19,
        )
        colors = np.array(
            [
                [255, 120,  50, 255],       # barrier              orange
                [255, 192, 203, 255],       # bicycle              pink
                [255, 255,   0, 255],       # bus                  yellow
                [  0, 150, 245, 255],       # car                  blue
                [  0, 255, 255, 255],       # construction_vehicle cyan
                [255, 127,   0, 255],       # motorcycle           dark orange
                [255,   0,   0, 255],       # pedestrian           red
                [255, 240, 150, 255],       # traffic_cone         light yellow
                [135,  60,   0, 255],       # trailer              brown
                [160,  32, 240, 255],       # truck                purple                
                [255,   0, 255, 255],       # driveable_surface    dark pink
                # [175,   0,  75, 255],       # other_flat           dark red
                [139, 137, 137, 255],
                [ 75,   0,  75, 255],       # sidewalk             dard purple
                [150, 240,  80, 255],       # terrain              light green          
                [230, 230, 250, 255],       # manmade              white
                [  0, 175,   0, 255],       # vegetation           green
                [  0, 255, 127, 255],       # ego car              dark cyan
                [255,  99,  71, 255],       # ego car
                [  0, 191, 255, 255],        # ego car
                # [0,0,0,0],
            ]
        ).astype(np.uint8)
        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors
        mlab.show()
    
    def destroy(self):
        # if not self.world.get_actor(self.sensor.id):
        self.sensor.destroy()
        
    # def visualize_data(self):
    #     if self.draw_process is not None:
    #         self.draw_process.kill()

    #     self.draw_process = Process(target=VoxelDetection.draw, args=(self.voxels,self.agent_id,))
    #     self.draw_process.start()
        
