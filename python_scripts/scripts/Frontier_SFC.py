import numpy as np
import matplotlib.pyplot as plt
import aq_library as aq_lib 
import obstacles as obs 

'''
Class for 1) Frontier Points generation, 2) Select the most promising frontier points, 3) Safe Flight Corridor for each selected frontier points. 
'''
class Ft_SFC(object):
    def __init__(self, ranges=None, obstacle_world=obs.FreeWorld(), pos=None, lidar=None, aq_func=aq_lib.info_gain, time=0,
                belief = None):
        self.ranges = ranges # map size 
        self.obstacle_world = obstacle_world
        self.pos = pos # current (x,y) position 
        self.lidar = lidar 
        self.aq_func = aq_func
        self.selected_fts = None 
        self.clustered_fts = None 
        self.time = time 
        self.GP = belief 
        self.ft_num = 5 # Number of selected frontiers 
    
    def gen_clustered_frontier(self):
        frontiers = self.lidar.frontier_detection(self.pos)
        clustered_frontiers = self.lidar.frontier_clustering(frontiers)
        self.clustered_fts = clustered_frontiers 
        # print(clustered_frontiers)

    def select_fts(self):
        if self.clustered_fts is None:
            self.gen_clustered_frontier()

        if self.ft_num > len(self.clustered_fts):
            self.ft_num = len(self.clustered_fts)
            self.select_fts = self.clustered_fts #Select the whole clustered frontier set 
        
        val_array = np.empty(len(self.clustered_fts))
        i = 0
        if(self.clustered_fts is not None):
            for x_ft in self.clustered_fts:
                if(self.aq_func==aq_lib.mves ):
                    #TODO: Fix the paramter setting. This leads to wrong MES acquisition function computation.
                    # maxes, locs, funcs = sample_max_vals(robot_model=sim_world, t=t, nK = 3, nFeatures = 200, visualize = False, obstacles=obslib.FreeWorld(), f_rew='mes'): 
                    val_array[i] = self.aq_func(time = self.time, xvals = x_ft, param= (self.max_val, self.max_locs, self.target), robot_model = self.GP)
                    i = i + 1
                else:
                    # print("Value: ", self.clustered_fts)
                    # x_ft = np.array(x_ft)
                    val_array[i] = self.aq_func(time = self.time, xvals = x_ft, robot_model = self.GP)
                    i = i + 1
            #Sort clustered frontiers with respect to the evaluation values. 
            # print(val_array)
            val_array, self.clustered_fts = (list(t) for t in zip(*sorted(zip(val_array, self.clustered_fts))))
            self.selected_fts = self.clustered_fts[0:self.ft_num]
            # print("Selection Called")
            # print(self.clustered_fts)
            # print(self.selected_fts)
            self.clustered_fts = np.array(self.clustered_fts)
        else:
            self.gen_clustered_frontier()
            self.select_fts()

    def get_clustered_frontier(self):
        if self.clustered_fts is not None:
            return self.clustered_fts   
        else:
            self.gen_clustered_frontier()
            return self.clustered_fts

    def get_selected_frontier(self):
        if self.selected_fts is not None:
            return self.selected_fts
        else:
            self.select_fts()
            return self.selected_fts

    def gen_SFC(self):
        if self.selected_fts is not None:
            self.lidar.selected_fts(self.selected_fts)
            SFC_vec = self.lidar.get_sfc(self.pos)
            print("SFC_vec", SFC_vec)
            return SFC_vec 
        else:
            self.select_fts()
            return SFC_vec

    # def get_SFC(self):
    #     if 
    # def visualize_Ft(self):

    # def visualize_SFC(self):

