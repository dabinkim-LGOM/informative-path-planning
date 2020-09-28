'''
Continuous Action MCTS with Tree Refinement Version 
'''

from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
from sklearn.neighbors import *
# from IPython.display import display
from scipy.stats import multivariate_normal
import numpy as np
import math
import os
import GPy as GPy
import dubins
import time
from itertools import chain
import aq_library as aqlib
# import glog as log
import logging as log
import copy
import random
# import gpmodel_library as gp_lib
# from continuous_traj import continuous_traj

from Environment import *
from Evaluation import *
from GPModel import *
import GridMap_library as sdflib
from Path_Generator import *
from MCTS import *
# from global_BO import ParticleSwarmOpt 
from cont_MCTS import Node, Tree

class Tree_TR(Tree):
    def __init__(self, ranges, obstacle_world, f_rew, f_aqu,  belief, pose, time, max_depth, max_rollout_depth, param, horizon_length, path_generator, frontier_size, c):
        self.ranges = ranges 
        self.path_generator = path_generator
        self.obstacle_world = obstacle_world
        self.max_depth = max_depth #Maximum tree depth?
        self.max_rollout_depth = max_rollout_depth
        # self.max_rollout_depth = max_rollout_depth
        self.param = param
        self.time = time
        self.f_rew = f_rew
        self.acquisition_function = f_aqu
        self.belief = belief 
        self.xvals = None 
        self.zvals = None 
        self.initial_pose = pose 
        self.root = Node(pose, parent = None, name = 'root', action = None, dense_path = None, zvals = None)  

        variance = 100.0 
        lengthscale = 3.0
        self.rGP_lengthscale = lengthscale
        self.rGP_variance = variance
        self.gp_kern = GPy.kern.RBF(input_dim = 2, lengthscale = lengthscale, variance = variance) # kernel of GP for reward 
        self.x_bound = horizon_length
        self.y_bound = horizon_length 
        self.num_action = frontier_size # Number of actions 
        self.c = c 


class Node_Set(object):
    '''
    Class which contains all node set with position coordinates. It is used for fast neighbor search of tree. And revise tree structure
    '''
    def __init__(self, leaf_num, neighbor_thres, tree):
        self.leaf_num = leaf_num
        self.neighbor_thres = neighbor_thres
        if tree is not None:
            self.node_list = self.get_leaf_nodes(tree)
            self.pos_list = self.convert_to_pos(self.node_list)
        else:
            self.node_list = []
            self.pos_list = []

    def get_leaf_nodes(self,tree):
        #Get all leaf nodes of tree structure
        leafs = [] 
        def __get_leaf_nodes(node):
            if node is not None:
                if len(node.children) ==0:
                    # leafs.append(np.array(node.pose[0], node.pose[1]))
                    leafs.append(node)
                for n in node.children:
                    __get_leaf_nodes(n)
        __get_leaf_nodes(tree.root)
        return leafs 

    def add_node(self, node):
 
        self.node_vec(node)
        self.build_kdtree()

    def convert_to_pos(self, node_list):
        pos_list = []
        for i, node in enumerate(node_list):
            pos_list.append(np.array(node.pose[0], node.pose[1]))
        return pos_list 

    def build_kdtree(self):
        #Convert node_vec to position value based kdtree
        if(len(self.node_list) != len(self.pos_list)):
            self.pos_list = self.convert_to_pos(self.node_list)
        self.kdtree = KDTree(self.pos_list, leaf_size=self.leaf_num)

    def get_neighbor(self, node):
        #From built kdtree, find nearest nodes which is under threshold. 
        try:
            node_idx = self.node_list.index(node)
        except expression as identifier:
            print("Node is not in node set")
            pass
        dist, ind = self.kdtree.query(self.pos_list[:node_idx], k=10)
        neighbor_list = []
        for i in range(len(dist)):
            if(dist[i] > self.neighbor_thres):
                break
            if(i>0):
                neighbor_list.append(self.node_list[ind[i]])
        return neighbor_list


