import numpy as np
import math
import grid_map_ipp_module as grid
import GPModel as GPlib

class sdf_map():
    def __init__(self, ranges, obsWorld, lidar, pos, length):
        self.ranges = ranges 
        self.obsWorld = obsWorld
        self.lidar = lidar 
        
        self.GPMap = None
        cur_pos = np.zeros(shape=(2,1))
        if(len(pos)==3):
            cur_pos[0], cur_pos[1] = pos[0], cur_pos[1]
        else:
            cur_pos = pos 

        self.sdf_map = grid.GridMapSDF(1.0, self.lidar, cur_pos, length) ##SDF map based on submap of belief map(from lidar object). Centered at pos, and dimension of length
        self.sdf_map.generate_SDF()

    def is_exist_obstacle(self):
        '''
        Check whether there is occupied voxel in the submap. SDF map will be generated if there is occupied voxel. 
        Return : bool, True: there is occupied voxel. False: if not. 
        '''
        

    def get_sdf_value(self, pos):
        return self.sdf_map.get_distance(pos)
    
    def get_sdf_gradient(self, pos):
        return self.sdf_map.get_gradient_value(pos)

    def generate_GP(self, obs_data):
        sdf_data = np.zeros(shape=(obs_data.size,1))
        for i in range(obs_data[0].size):
            # print(obs_data.size)
            sdf_data[i] = self.get_sdf_value(obs_data[i])
        lengthscale = 1.0
        variance = 100.0
        print(obs_data)
        print('SDF')
        print(sdf_data)
        self.GPMap = GPlib.GPModel(lengthscale=lengthscale, variance=variance)
        self.GPMap.set_data(obs_data, sdf_data)

    def add_data(self, obs_data, sdf_data):
        if self.GPMap == None:
            self.generate_GP(obs_data)
        else:
            self.GPMap.add_data(obs_data, sdf_data)
    
    def query_sdf(self, query_pos):
        return self.GPMap.predict_value(query_pos)

    def visualize_sdf(self):
        self.GPMap.visualize_model(self.ranges[1], self.ranges[3])


if __name__ == "__main__":
    xx = np.load('xdata.npy')