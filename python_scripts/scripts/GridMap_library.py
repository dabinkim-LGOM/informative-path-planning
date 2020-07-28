import numpy as np
import math
import grid_map_ipp_module as grid
import GPModel as GPlib

class sdf_map():
    def __init__(self, ranges, obsWorld):
        self.ranges = ranges 
        self.obsWorld = obsWorld
    
    def initialize(self):
        sdf_map = grid.GridMap_SDF(1.0, self.ranges[1], self.ranges[3], 5, self.obsWorld)
        sdf_map.generate_SDF()
    

    # def generate_GP(self, data):
