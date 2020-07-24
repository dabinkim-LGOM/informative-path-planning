import numpy as np

class sdf_map():
    def __init__(self, ranges, obsWorld):
        self.ranges = ranges 
        self.obsWorld = obsWorld
    
    def generate(self):
        sdf_map = grid.GridMap_SDF(1.0, self.ranges[0], self.ranges[2], 5, np_centers)
        sdf_map.generate_SDF()