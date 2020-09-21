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
from global_BO import ParticleSwarmOpt 

from MCTS import *

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
        self.children = children 

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
    def __init__(self, ranges, obstacle_world, f_rew, f_aqu,  belief, pose, t, max_depth, param):
        self.ranges = ranges 
        # self.path_generator = path_generator
        self.obstacle_world = obstacle_world
        self.max_depth = max_depth #Maximum tree depth?
        # self.max_rollout_depth = max_rollout_depth
        self.param = param
        self.t = t
        self.f_rew = f_rew
        self.aquisition_function = f_aqu
        # self.turning_radius = turning_radius
        # self.c = c
        # self.gradient_on = gradient_on
        # self.grad_step = grad_step
        self.belief = belief 

        self.root = Node(pose, parent = None, name = 'root', action = None, dense_path = None, zvals = None)  
        #self.build_action_children(self.root) 

    def get_best_child(self):
        return self.root.children[np.argmax([node.nqueries for node in self.root.children])]

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
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'exp_improve':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive':
                # param = sample_max_vals(belief, t=self.t, nK=int(self.param[0]))
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)#(param, self.param[1]))
            elif self.f_rew == 'naive_value':
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief, param = self.param)
            else:
                r = self.aquisition_function(time = self.t, xvals = xobs, robot_model = belief)
            
            cumul_reward += r
            cur_depth += 1
        return cumul_reward 

    def get_reward(self, sequence):
        '''Evaluate the sequence to get the reward, defined by the percentage of entropy reduction'''
        # The process is iterated until the last node of the rollout sequence is reached 
        # and the total information gain is determined by subtracting the entropies 
        # of the initial and final belief space.
        # reward = infogain / Hinit (joint entropy of current state of the mission)
        sim_world = self.belief
        samples = []
        obs = []
        for seq in sequence:
            samples.append(seq)
        obs = np.array(samples)
        # print "Get reward samples: ", samples 
            # print obs.size
        if(len(obs)==0):
            print("Observation set is empty", current_node)
                # continue 
        xobs = np.vstack([obs[:,0], obs[:,1]]).T
        # obs = list(chain.from_iterable(samples))
        # print "Obs", xobs
        if(self.aquisition_function==aqlib.mves ):
            #TODO: Fix the paramter setting. This leads to wrong MES acquisition function computation.
            # maxes, locs, funcs = sample_max_vals(robot_model=sim_world, t=t, nK = 3, nFeatures = 200, visualize = False, obstacles=obslib.FreeWorld(), f_rew='mes'): 
            return self.aquisition_function(time = self.t, xvals = xobs, param= (self.max_val, self.max_locs, self.target), robot_model = sim_world)
        else:
            return self.aquisition_function(time = self.t, xvals = xobs, robot_model = sim_world)

    def get_next_child(self, current_node):
        vals = {}
        # e_d = 0.5 * (1.0 - (3.0/10.0*(self.max_depth - current_node.depth)))
        e_d = 0.5 * (1.0 - (3.0/(10.0*(self.max_depth - current_node.depth))))
        for i, child in enumerate(current_node.children):
            #print "Considering child:", child.name, "with queries:", child.nqueries
            if child.nqueries == 0:
                return child
            # vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt((float(current_node.nqueries) ** e_d)/float(child.nqueries)) 
            vals[child] = child.reward/float(child.nqueries) + self.c * np.sqrt(np.log(float(current_node.nqueries))/float(child.nqueries)) 
        # Return the max node, or a random node if the value is equal
        return random.choice([key for key in vals.keys() if vals[key] == max(vals.values())])
        
    def build_action_children(self, parent):
        # print(self.path_generator.get_path_set(parent.pose))
        # print(self.path_generator)
        actions, dense_paths = self.path_generator.get_path_set(parent.pose)

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
                parent.add_children(Node(pose = parent.pose, 
                                        parent = parent, 
                                        name = parent.name + '_action' + str(i), 
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

class CBTS(MCTS):
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
                path_generator, aquisition_function, f_rew, time):
        super(CBTS, self): __init__(ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
                path_generator, aquisition_function, f_rew, time )
        self.ranges = ranges
        self.time = time 
        self.obstacle_world = obstacle_world
        self.belief = belief 
        self.acquisition_function = aquisition_function
        self.f_rew = f_rew 
        self.computation_budget = computation_budget
        self.initial_pose = initial_pose
        self.path_generator = path_generator 
        # self.optimizer = ParticleSwarmOpt
        self.x_bound, self.y_bound = 10.0, 10.0 
        self.num_action = frontier_size # Number of actions 

        self.gp_kern = GPy.kern.RBF(input_dim = self.dim, lengthscale = lengthscale, variance = variance) # kernel of GP for reward 

    def get_actions(self):
        # self.tree = Tree(self.ranges, self.obstacle_world, self.f_rew, self.acquisition_function, self.belief, self.initial_pose, self.path_generator, self.time, 
        #                 max_depth = self.max_depth, max_rollout_depth= self.max_rollout_depth, turning_radius = self.turning_radius, 
        #                 param = param, c = self.c, gradient_on=self.gradient_on, grad_step=self.grad_step)

        self.tree = self.initialize_tree()
        time_start = time.clock()

        # self.sdf_map = self.generate_sdfmap(self.cp)
        print("Current timestep : ", self.t)

        # if self.sdf_map.is_exist_obstacle():
        #     print("Here")
        #     self.sdf_map.train_sdfmap(train_num=10, cp = self.cp)
        iteration = 0
        while time.clock() - time_start < self.budget:
            current_node = self.tree_policy() #Find maximum UCT node (which is leaf node)
            iteration +=1
            # #Update SDF map
            # if self.sdf_map is not None:
            #     print("Here")
                # self.update_sdfmap(current_node)

            sequence = self.rollout_policy(current_node, self.budget) #Add node
            
            reward = self.get_reward(sequence)
            # print("cur_reward : " + str(reward))
            value_grad = self.get_value_grad(current_node, sequence, reward)
            # if(len(self.tree[sequence[0]])==4):
            #     self.tree[sequence[0]] = (self.tree[sequence[0]][0],self.tree[sequence[0]][1], self.tree[sequence[0]][2], self.tree[sequence[0]][3], value_grad )
            
            self.backprop(reward, sequence, value_grad)
            ###TODO: After Finish Build functions for update
            # if(self.gradient_on):
            #     self.update_action(reward, sequence, None)
            # else:
            #     self.backprop(reward, sequence, value_grad)
            
        # self.visualize_tree()
        print("Rollout number : ", iteration)
        print("Time spent : ", time.clock() - time_start)

        best_sequence, cost = self.get_best_child()


        # update_ver = self.update_action(self.tree[best_sequence])
        if(self.gradient_on == True):
            update_ver = self.update_action(self.tree[best_sequence])
            return update_ver[0], cost
        else:
            return self.tree[best_sequence][0], cost

    def initialize_tree(self):
        # '''Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world'''
        # tree = 
        #(pose, number of queries)
        tree['root'] = (self.initial_pose, 0)

        # actions, _ = self.path_generator.get_path_set(self.cp)
        # feas_actions = self.collision_check(actions)

        for action, samples in feas_actions.items():
            #(samples, cost, reward, number of times queried)
            cost = np.sqrt((self.cp[0]-samples[-1][0])**2 + (self.cp[1]-samples[-1][1])**2)
            tree['child '+ str(action)] = (samples, cost, 0, 0)
        return tree


    def tree_policy(self):
        '''Implements the UCB policy with continuous action to select the child to expand and forward simulate'''
        # According to Arora:
        #avg_r average reward of all rollouts that have passed through node n
        #c_p some constant , 0.1 in literature
        #N number of times parent has been evaluated
        #n number of time node n has been evaluated
        #ucb = avg_r + c_p*np.sqrt(2*np.log(N)/n)
        leaf_eval = {}
        for i in xrange(self.frontier_size):
            node = 'child '+ str(i)
            if(node in self.tree): #If 'node' string key value is in current tree. 
                leaf_eval[node] = self.tree[node][2] + self.c*np.sqrt(2*(np.log(self.tree['root'][1]))/(self.tree[node][3]))
        return max(leaf_eval, key=leaf_eval.get)

    def action_selection_BO(self, num_BO, xvals, zvals):
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
        if xvals is None:
            xvals = np.random.rand(num_prior,2)*[self.x_bound, self.y_bound] + self.initial_pose 
            zvals = self.aquisition_function(time = self.time, xvals = xvals, robot_model = self.belief)

        for i in ranges(num_BO):
            reward_GP = GPy.models.SparseGPRegression(np.array(xvals), np.array(zvals), self.kern)
            self.optimizer = ParticleSwarmOpt(self.time, self.obstacle_world, reward_GP, self.acquisition_function, pose, self.x_bound, self.y_bound)
            best_action, reward = self.optimizer.optimization()
            
            #Augment dataset 
            xvals = np.vstack([xvals, best_action])
            zvals = np.vstack([zvals, reward])

        return max_action, xvals, zvals

    def get_best_child(self):
        '''Query the tree for the best child in the actions'''
        best = -1000
        best_child = None
        for i in xrange(self.frontier_size):
            if('child '+ str(i) in self.tree):
                r = self.tree['child '+ str(i)][2]
                if r > best:
                    best = r
                    best_child = 'child '+ str(i)
        return best_child, 0

    # def run_optimizer(self, pose):

    #     best_sol, best_val = self.optimizer.optimization()


