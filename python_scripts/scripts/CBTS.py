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
from baseline.global_BO import ParticleSwarmOpt 

import MCTS as mc

'''
Baseline code for Continuous Belief Tree Search using PSO as global optimization for bayesian optimization 
IROS18 "Continuous State-Action-Observation POMDPs for Trajectory Planning with Bayesian Optimization" 
'''

class Node_C(object):
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


class Tree_C(object):
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
        self.root = Node_C(pose, parent = None, name = 'root', action = None, dense_path = None, zvals = None)  

        variance = 100.0 
        lengthscale = 3.0
        self.rGP_lengthscale = lengthscale
        self.rGP_variance = variance
        self.gp_kern = GPy.kern.RBF(input_dim = 2, lengthscale = lengthscale, variance = variance) # kernel of GP for reward 
        self.x_bound, self.y_bound = horizon_length, horizon_length 
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
            
            selected_action =random.choice(actions) 

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

    # def get_reward(self, sequence):
    #     '''Evaluate the sequence to get the reward, defined by the percentage of entropy reduction'''
    #     # The process is iterated until the last node of the rollout sequence is reached 
    #     # and the total information gain is determined by subtracting the entropies 
    #     # of the initial and final belief space.
    #     # reward = infogain / Hinit (joint entropy of current state of the mission)
    #     sim_world = self.belief
    #     samples = []
    #     obs = []
    #     for seq in sequence:
    #         samples.append(seq)
    #     obs = np.array(samples)
    #     # print "Get reward samples: ", samples 
    #         # print obs.size
    #     if(len(obs)==0):
    #         print("Observation set is empty", current_node)
    #             # continue 
    #     xobs = np.vstack([obs[:,0], obs[:,1]]).T
    #     # obs = list(chain.from_iterable(samples))
    #     # print "Obs", xobs
    #     if(self.aquisition_function==aqlib.mves ):
    #         #TODO: Fix the paramter setting. This leads to wrong MES acquisition function computation.
    #         # maxes, locs, funcs = sample_max_vals(robot_model=sim_world, t=t, nK = 3, nFeatures = 200, visualize = False, obstacles=obslib.FreeWorld(), f_rew='mes'): 
    #         return self.aquisition_function(time = self.t, xvals = xobs, param= (self.max_val, self.max_locs, self.target), robot_model = sim_world)
    #     else:
    #         return self.aquisition_function(time = self.t, xvals = xobs, robot_model = sim_world)

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
            
        while(parent.depth <=self.max_depth):
            if(parent.children==None):
                parent.children = []
            print("num action", self.num_action)
            print("children", parent.children)
            if(len(parent.children) <self.num_action):

                goal_vec = np.empty((0,3))
                # for i in xrange(self.num_action):
                goal, xvals, zvals = self.action_selection_BO(parent.pose ,5, self.xvals, self.zvals)
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
        # return cur_node.children[np.argmax([node.nqueries for node in cur_node.children])]


    def action_selection_BO(self, cur_pose, num_BO, xvals, zvals):
        '''
        Run Bayesian Optimization for action selection in continuous domain 
        Parameter: 
            num_BO - Number of iterations for BO 
            xvals, zvals - Dataset 
        Return:
            max_action - 
            xvals, zvals - Augmented Dataset 

        1. First, if dataset is empty, gather pre-defined number of prior samples to train GP model 
        2. For fixed # of iterations, run BO with PSO in inner optimization
        '''
        num_prior = 10
        if xvals is None:
            # print(self.initial_pose)
            xvals = np.random.rand(num_prior,2)*[self.x_bound, self.y_bound] + self.initial_pose[0:1].transpose()
            zvals = np.empty(shape=(num_prior,1))
            
            for i, values in enumerate(xvals):
                aq_val = self.acquisition_function(time = self.time, xvals = values, robot_model = self.belief)
                zvals[i] = aq_val

        best_action = cur_pose
        max_reward = -1e5
        # print(xvals)
        # print(zvals)

        for i in range(num_BO):
            # reward_GP = GPy.models.SparseGPRegression(np.array(xvals), np.array(zvals), self.gp_kern)
            reward_GP = GPModel(lengthscale=self.rGP_lengthscale, variance=self.rGP_variance)
            reward_GP.set_data(np.array(xvals), np.array(zvals))
            self.optimizer = ParticleSwarmOpt(self.ranges, self.time, self.obstacle_world, reward_GP, self.acquisition_function, cur_pose, self.x_bound, self.y_bound)
            
            cur_pose, reward = self.optimizer.optimization()
            if(reward > max_reward):
                best_action = cur_pose
                max_reward = reward 
            #Augment dataset 
            xvals = np.vstack([xvals, cur_pose])
            zvals = np.vstack([zvals, reward])
        
        return best_action, xvals, zvals

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

class CBTS(object):
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, horizon_length, frontier_size,
                path_generator, aquisition_function, f_rew, time):
        # super(CBTS, self).__init__(ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, horizon_length, frontier_size,
        #         path_generator, aquisition_function, f_rew, time )
        self.ranges = ranges
        self.time = time 
        self.obstacle_world = obstacle_world
        self.belief = belief 
        self.acquisition_function = aquisition_function
        self.f_rew = f_rew 
        self.computation_budget = computation_budget
        self.initial_pose = initial_pose
        self.path_generator = path_generator 
        self.horizon_length = horizon_length
        self.max_depth = max_depth
        self.max_rollout_depth = max_rollout_depth
        # self.param = param 

        self.c = 0.1 #Parameter for UCT function 
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
        # self.optimizer = ParticleSwarmOpt


    def get_actions(self):
        # self.tree = Tree(self.ranges, self.obstacle_world, self.f_rew, self.acquisition_function, self.belief, self.initial_pose, self.path_generator, self.time, 
        #                 max_depth = self.max_depth, max_rollout_depth= self.max_rollout_depth, turning_radius = self.turning_radius, 
        #                 param = param, c = self.c, gradient_on=self.gradient_on, grad_step=self.grad_step)

        # randomly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs, self.target  = aqlib.sample_max_vals(self.belief, t = self.time, visualize=True)
            param = (self.max_val, self.max_locs, self.target)
        elif self.f_rew == 'exp_improve':
            param = [self.current_max]
        elif self.f_rew == 'naive' or self.f_rew == 'naive_value':
            self.max_val, self.max_locs, self.target  = aqlib.sample_max_vals(self.belief, t= self.time, nK=int(self.aq_param[0]), visualize=True, f_rew=self.f_rew)
            param = ((self.max_val, self.max_locs, self.target), self.aq_param[1])
        else:
            param = None

        self.tree = Tree_C(self.ranges, self.obstacle_world, self.f_rew, self.acquisition_function, self.belief, np.asarray(self.initial_pose), self.time, self.max_depth, self.max_rollout_depth, param, 
                        self.horizon_length, self.path_generator, self.horizon_length, self.c)
        current_node = self.tree.root

        time_start = time.clock()

        print("Current timestep : ", self.time)

        iteration = 0

        self.tree.build_initial_action(self.tree.root)
        while time.clock() - time_start < self.computation_budget:

            current_node = self.tree.tree_policy(current_node)
            # current_node = self.tree.get_next_leaf_node(current_node) #Find maximum UCT node (which is leaf node)
            iteration +=1

            reward = self.tree.rollout(current_node, self.belief) #Add node
            print("Rollout occurs")
            self.tree.backprop(current_node, reward)
            
            current_node = self.tree.root
            
            
        self.tree.print_tree()
        print("Rollout number : ", iteration)
        print("Time spent : ", time.clock() - time_start)

        best_child, reward = self.tree.get_best_child()


        # update_ver = self.update_action(self.tree[best_sequence])
        return best_child.action, best_child.dense_path, reward 


    # def tree_policy(self):
    #     '''Implements the UCB policy with continuous action to select the child to expand and forward simulate'''
    #     # According to Arora:
    #     #avg_r average reward of all rollouts that have passed through node n
    #     #c_p some constant , 0.1 in literature
    #     #N number of times parent has been evaluated
    #     #n number of time node n has been evaluated
    #     #ucb = avg_r + c_p*np.sqrt(2*np.log(N)/n)
    #     leaf_eval = {}
    #     for i in xrange(self.frontier_size):
    #         node = 'child '+ str(i)
    #         if(node in self.tree): #If 'node' string key value is in current tree. 
    #             leaf_eval[node] = self.tree[node][2] + self.c*np.sqrt(2*(np.log(self.tree['root'][1]))/(self.tree[node][3]))
    #     return max(leaf_eval, key=leaf_eval.get)



    # def run_optimizer(self, pose):

    #     best_sol, best_val = self.optimizer.optimization()


