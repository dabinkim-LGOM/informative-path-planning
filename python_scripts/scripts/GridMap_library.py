import numpy as np
import math
import GPModel as GPlib

class sdf_map():
    def __init__(self, ranges, obsWorld, lidar):
        self.ranges = ranges 
        self.obsWorld = obsWorld
        self.lidar = lidar 
    
    def initialize(self):
        sdf_map = grid.GridMap_SDF(1.0, self.lidar)
        sdf_map.generate_SDF()
    

    def generate_GP(self, obs_data, sensor_data):
        lengthscale = 1.0
        variance = 1.0

        GPMap = GPlib.GPModel(lengthscale=lengthscale, variance=variance)
        GPMap.set_data(obs_data, sensor_data)

