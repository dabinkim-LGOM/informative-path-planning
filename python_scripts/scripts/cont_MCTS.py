from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
from matplotlib import cm
from sklearn import mixture
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
# from global_BO import ParticleSwarmOpt 


class Node(object):
    def __init__(self, pose, parent, name, action = None, dense_path = None, zvals = None):
        self.pose = pose
        self.name = name
        self.zvals = zvals 
        self.reward = 0.0
        self.nqueries = 0

        self.parent = parent 
        self.children = None

        self.action = action 
        self.dense_path = dense_path 
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth +1 

    def add_children(self, child_node):
        if self.children is None:
            self.children = [] 
        self.children.append(child_node)
    
    def remove_children(self, child_node):
        if child_node in self.children: self.children.remove(child_node)
    
    def print_self(self):
        print(self.name)


class Tree(object):
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

        #self.build_action_children(self.root) 

    def get_best_child(self):
        if len(self.root.children)==1:
            # print("length 1")
            # print(self.root.children[0])
            return self.root.children[0], self.root.children[0].reward
        else:
            # print("length more than 1")
            # print(self.root.children)
            max_nquery = 0
            max_idx = 0
            for i in range(len(self.root.children)):
                nquery = self.root.children[i].nqueries
                if nquery > max_nquery:
                    max_nquery = nquery
                    max_idx = i
            # max_idx = np.argmax([node.nqueries for node in self.root.children])
            return self.root.children[max_idx], self.root.children[max_idx].reward 
            
        # print(self.root.children)
        # if len(self.root.children) >1:
        #     return self.root.children[np.argmax([node.nqueries for node in self.root.children])]
        # else:
        #     return self.root.children[0]

    def backprop(self, leaf_node, reward):
        if leaf_node.parent is None:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            #print "Calling backprop on:",
            #leaf_node.print_self()
            #print "nqueries:", leaf_node.nqueries, "reward:", leaf_node.reward
            return
        else:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            #print "Calling backprop on:",
            #leaf_node.print_self()
            #print "nqueries:", leaf_node.nqueries, "reward:", leaf_node.reward
            self.backprop(leaf_node.parent, reward)
            return
    
    
    def rollout(self, current_node, belief):
        cur_depth = current_node.depth
        tmp_depth = cur_depth 
        cur_pose = current_node.pose 
        cumul_reward = 0.0 
        while cur_depth <= tmp_depth + self.max_rollout_depth:
            actions, dense_paths = self.path_generator.get_path_set(cur_pose)
            
            selected_action = random.choice(actions) 

            if(len(selected_action) == 0):
                return cumul_reward
            # print(selected_action)
            cur_pose = selected_action[-1]
            
            obs = np.array(cur_pose)
            xobs = np.vstack([obs[0], obs[1]]).T

            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'exp_improve':
                r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive':
                # param = sample_max_vals(belief, t=self.t, nK=int(self.param[0]))
                r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)#(param, self.param[1]))
            elif self.f_rew == 'naive_value':
                r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            else:
                r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief)
            
            cumul_reward += r
            cur_depth += 1
        return cumul_reward 


    def get_next_child(self, current_node):
        vals = {}
        # e_d = 0.5 * (1.0 - (3.0/10.0*(self.max_depth - current_node.depth)))
        e_d = 0.5 * (1.0 - (3.0/(10.0*(self.max_depth - current_node.depth))))
        # print(current_node.children)
        for i, child in enumerate(current_node.children):
            #print "Considering child:", child.name, "with queries:", child.nqueries
            if child.nqueries == 0:
                return child
            # vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt((float(current_node.nqueries) ** e_d)/float(child.nqueries)) 
            vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt(np.log(float(current_node.nqueries))/float(child.nqueries)) 
        # Return the max node, or a random node if the value is equal
        # print(vals.keys())
        # print(vals.values())
        return random.choice([key for key in vals.keys() if vals[key] == max(vals.values())])
    
    #Find leaf node based on UCT criteria
    def get_next_leaf_node(self, current_node):
        if(current_node.depth == self.max_depth or current_node.children is None):
            return current_node
        else:
            child_node = self.get_next_child(current_node)
            return self.get_next_leaf_node(child_node)

    def tree_policy(self, parent):
        # print(self.path_generator.get_path_set(parent.pose))
        # print(self.path_generator)
        # actions, dense_paths = self.path_generator.get_path_set(parent.pose)
        cur_node = parent 
        num_BO = 3 
        while(parent.depth <self.max_depth):
            if(parent.children==None):
                parent.children = []
            print("num action", self.num_action)
            print("children", parent.children)
            if(len(parent.children) <self.num_action):

                goal_vec = np.empty((0,3))
                # for i in xrange(self.num_action):
                goal, xvals, zvals = self.action_selection_BO(parent.pose , num_BO, self.xvals, self.zvals)
                goal_vec = np.vstack([goal_vec, np.append(goal, self.initial_pose[2])])
                self.xvals, self.zvals = xvals, zvals
                # print('goal_vec', goal_vec)
                actions, dense_paths = self.path_generator.get_path_set_w_goals(parent.pose, goal_vec)
            # print "Actions: ", actions
                free_actions, free_dense_paths = self.collision_check(actions, dense_paths)
            # print "Action set: ", free_actions 
            # actions = self.path_generator.get_path_set(parent.pose)
            # dense_paths = [0]
                if len(actions) == 0:
                    print("No actions!")
                    return
                
                # print "Creating children for:", parent.name
                for i, action in enumerate(free_actions.keys()):
                    # print "Action:", parent.name + '_action' + str(i)
                    if len(free_actions[action])!=0:
                        # print free_actions[action]
                        new_node = Node_C(pose = parent.pose, 
                                                parent = parent, 
                                                name = parent.name + '_action' + str(i), 
                                                action = free_actions[action], 
                                                dense_path = free_dense_paths[action],
                                                zvals = None)
                        parent.add_children(new_node)
                        # print("Adding next child: ", parent.name + '_action' + str(i))
                        return new_node 
            else:
                # Return best child from current parent node 
                parent = parent.children[np.argmax([node.nqueries for node in parent.children])]
                # return parent.children[np.argmax([node.nqueries for node in parent.children])]
        if(parent.depth ==self.max_depth):
            return parent
        # return cur_node.children[np.argmax([node.nqueries for node in cur_node.children])]



    def build_initial_action(self, parent):

        self.path_generator.fs = 3 #Number of initial action 
        actions, dense_paths = self.path_generator.get_path_set(parent.pose)

        free_actions, free_dense_paths = self.collision_check(actions, dense_paths)

        for i, action in enumerate(free_actions.keys()):
            # print "Action:", parent.name + '_action' + str(i)
            if len(free_actions[action])!=0:
                # print free_actions[action]
                parent.add_children(Node_C(pose = parent.pose, 
                                        parent = parent, 
                                        name = parent.name + '_action_initial' + str(i), 
                                        action = free_actions[action], 
                                        dense_path = free_dense_paths[action],
                                        zvals = None))
                # print("Adding next child: ", parent.name + '_action' + str(i))
    def collision_check(self, path_dict, path_dense_dict):
        free_paths = {}
        dense_free_paths = {}
        for key,path in path_dict.items():
            is_collision = 0
            for pt in path:
                if(self.obstacle_world.in_obstacle(pt, 3.0)):
                    is_collision = 1
            if(is_collision == 0):
                free_paths[key] = path
                dense_free_paths[key] = path_dense_dict[key]
        
        return free_paths, dense_free_paths
    

    def print_tree(self):
        counter = self.print_helper(self.root)
        print("# nodes in tree:", counter)

    def print_helper(self, cur_node):
        if cur_node.children is None:
            #cur_node.print_self()
            #print cur_node.name
            return 1
        else:
            #cur_node.print_self()
            #print "\n"
            counter = 0
            for child in cur_node.children:
                counter += self.print_helper(child)
            return counter


class conti_MCTS(MCTS):
    '''MCTS for continuous action'''
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, turning_radius, frontier_size,
                path_generator, aquisition_function, f_rew, time, gradient_on, grad_step, lidar, SFC):
        # Call the constructor of the super class
        super(cMCTS, self).__init__(ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
                                    path_generator, aquisition_function, f_rew, time, gradient_on, grad_step, lidar, SFC)
        self.tree_type = 'dpw'
        # Tree type is dpw 
        # self.aq_param = aq_param
        self.GP = belief
        self.f_rew = f_rew
        self.max_rollout_depth = max_rollout_depth
        self.comp_budget = computation_budget
        self.cp = initial_pose
        self.max_depth = max_depth
        self.frontier_size = frontier_size
        self.path_generator = path_generator
        self.obstacle_world = obstacle_world
        # self.default_path_generator = Path_Generator(frontier_size, )
        self.spent = 0
        self.tree = None
        self.c = 0.1 #Parameter for UCT function 
        self.aquisition_function = aquisition_function
        self.turning_radius = turning_radius
        self.t = time
        self.gradient_on = gradient_on
        self.grad_step = grad_step
        self.lidar = lidar
        self.sdf_map = None
        self.SFC = SFC #SFC Bounding box for optimization 

        # The differnt constatns use logarthmic vs polynomical exploriation
        if self.f_rew == 'mean':
            # if self.tree_type == 'belief':
            #     self.c = 1000
            # elif self.tree_type == 'dpw':
            self.c = 5000
        elif self.f_rew == 'exp_improve':
            self.c = 200
        elif self.f_rew == 'mes':
            # if self.tree_type == 'belief':
            #     self.c = 1.0 / np.sqrt(2.0)
            # elif self.tree_type == 'dpw':
                # self.c = 1.0 / np.sqrt(2.0)
            self.c = 1.0
                # self.c = 5.0
        else:
            self.c = 1.0

    def choose_trajectory(self):
        #Main function loop which makes the tree and selects the best child
        #Output: path to take, cost of that path
        print("Current Time: ", self.t)
        # randomly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t = self.t, visualize=True)
            param = (self.max_val, self.max_locs, self.target)
        elif self.f_rew == 'exp_improve':
            param = [self.current_max]
        elif self.f_rew == 'naive' or self.f_rew == 'naive_value':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t= self.t, nK=int(self.aq_param[0]), visualize=True, f_rew=self.f_rew)
            param = ((self.max_val, self.max_locs, self.target), self.aq_param[1])
        else:
            param = None

        # initialize tree
        if self.tree_type == 'dpw':
            self.tree = Tree(self.ranges, self.obstacle_world, self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, self.t, max_depth = self.max_depth,
                            max_rollout_depth= self.max_rollout_depth, turning_radius = self.turning_radius, param = param, c = self.c, gradient_on=self.gradient_on, grad_step=self.grad_step)
        # elif self.tree_type == 'belief':
            # self.tree = BeliefTree(self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, t, depth = self.rl, param = param, c = self.c)
        else:
            raise ValueError('Tree type must be one of either \'dpw\' or \'belief\'')
        #self.tree.get_next_leaf()
        #print self.tree.root.children[0].children

        time_start = time.clock()            
        # while we still have time to compute, generate the tree
        i = 0
        while time.clock() - time_start < self.comp_budget:#i < self.comp_budget:
            i += 1
            gp = copy.copy(self.GP)
            self.tree.get_next_leaf(gp)

            if True:
                gp = copy.copy(self.GP)
        time_end = time.clock()
        print("Rollouts completed in", str(time_end - time_start) +  "s")
        print("Number of rollouts:", i)
        self.tree.print_tree()

        print([(node.nqueries, node.reward/(node.nqueries+0.1)) for node in self.tree.root.children])

        #Executing the first action 
        best_child = self.tree.root.children[np.argmax([node.nqueries for node in self.tree.root.children])]
        # best_child = random.choice([node for node in self.tree.root.children if node.nqueries == max([n.nqueries for n in self.tree.root.children])])
        all_vals = {}
        for i, child in enumerate(self.tree.root.children):
            all_vals[i] = child.reward / (float(child.nqueries)+0.1)
            # print(str(i) + " is " + str(all_vals[i]))

        paths, dense_paths = self.path_generator.get_path_set(self.cp)

        # return best_child.action, best_child.dense_path, best_child.reward/(float(best_child.nqueries)+1.0), paths, all_vals, self.max_locs, self.max_val, self.target

        return best_child.action, best_child.dense_path, best_child.reward/(float(best_child.nqueries)+1.0) 


        # get the best action to take with most promising futures, base best on whether to
        # consider cost
        #best_sequence, best_val, all_vals = self.get_best_child()

        #Document the information
        #print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        #logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))
        #np.save('./figures/' + self.f_rew + '/tree_' + str(t) + '.npy', self.tree)
        #return self.tree[best_sequence][0], self.tree[best_sequence][1], best_val, paths, all_vals, self.max_locs, self.max_val

