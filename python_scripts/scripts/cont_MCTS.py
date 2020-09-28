'''
Continuous Action MCTS without Tree Refinement.
Tree structure is divided into conti-action depth and discrete-action depth. 
Value gradient update is used for continuous update in action space. 
Double Progressive Widening is adapted into only discrete-action depth. 
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


class Node(object):
    def __init__(self, pose, parent, name, action = None, dense_path = None, is_cont = False):
        #is_cont : is this node is a element of continuous-action node set?
        self.pose = pose
        self.name = name
        self.zvals = zvals 
        self.reward = 0.0
        self.nqueries = 0

        self.parent = parent 
        self.children = None


        self.max_action = 5 # Default number 
        self.action = action    
        self.dense_path = dense_path 
        self.is_cont = is_cont # is cont = true: No dpw, false: dpw
        # if self.is_cont:
        #     self.max_action = 

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth +1 

    def set_max_action(self, max_action, alpha):
        if self.is_cont:
            self.max_action = max_action
        else:
            self.max_action = math.floor(math.log(max_action,alpha)) # Based on progressive widening 

    def add_children(self, child_node):
        if self.children is None:
            self.children = [] 
        self.children.append(child_node)
    
    def remove_children(self, child_node):
        if child_node in self.children: self.children.remove(child_node)
    
    def print_self(self):
        print(self.name)


class Tree(object):
    def __init__(self, ranges, obstacle_world, f_rew, f_aqu,  belief, pose, time, max_depth, max_rollout_depth, param, horizon_length, path_generator, frontier_size, c, depth_conti):
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
        self.horizon_length = horizon_length
        self.belief = belief 
        self.xvals = None 
        self.zvals = None 
        self.initial_pose = pose 
        self.depth_conti = depth_conti
        self.root = Node(pose, parent = None, name = 'root', action = None, dense_path = None, zvals = None, is_cont=True)  

        self.num_action = frontier_size # Number of actions 
        self.c = c 

        #self.build_action_children(self.root) 

    def get_best_child(self):
        # if len(self.root.children)==1:
        #     return self.root.children[0], self.root.children[0].reward
        # else:
        max_nquery = 0
        max_idx = 0
        for i in range(len(self.root.children)):
            nquery = self.root.children[i].nqueries
            if nquery > max_nquery:
                max_nquery = nquery
                max_idx = i
        # max_idx = np.argmax([node.nqueries for node in self.root.children])
        return self.root.children[max_idx], self.root.children[max_idx].reward 
            
    def backprop(self, leaf_node, xobs, reward):
        # Backpropagate reward values to parent sequence 
        # At leaf node, xobs is given, and reward is computed based on its current belief function 
        # During recusirve call of the function, 'None' type xobs is passed therefore, reward is not redundanlty computed. 
        if xobs is not None: 
            if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
                reward += self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'exp_improve':
                reward += self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            elif self.f_rew == 'naive':
                # param = sample_max_vals(belief, t=self.t, nK=int(self.param[0]))
                reward += self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)#(param, self.param[1]))
            elif self.f_rew == 'naive_value':
                reward += self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            else:
                reward += self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief)

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
            self.backprop(leaf_node.parent, None, reward)
            return
    
    
    def rollout(self, current_node):
        '''
        Rollout function returns measurment observation positions from current node with random policy.
        This gives samples rather than rewards
        '''
        cur_depth = current_node.depth
        tmp_depth = cur_depth 
        cur_pose = current_node.pose 
        cumul_reward = 0.0 
        cumul_measruement = np.empty(shape=(0,2))
        while cur_depth <= tmp_depth + self.max_rollout_depth:
            actions, dense_paths = self.path_generator.get_path_set(cur_pose)
            
            selected_action = random.choice(actions) 

            if(len(selected_action) == 0):
                return cumul_reward
            # print(selected_action)
            cur_pose = selected_action[-1]
            
            obs = np.array(cur_pose)
            xobs = np.vstack([obs[0], obs[1]]).T
            cumul_measruement = np.vstack(cumul_measruement, xobs)
            # zobs = self.belief.predict_value(xobs)

            # if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            #     r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            # elif self.f_rew == 'exp_improve':
            #     r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            # elif self.f_rew == 'naive':
            #     # param = sample_max_vals(belief, t=self.t, nK=int(self.param[0]))
            #     r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)#(param, self.param[1]))
            # elif self.f_rew == 'naive_value':
            #     r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
            # else:
            #     r = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief)
            
            # cumul_reward += r
            
            cur_depth += 1
        return cumul_measruement 


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
    #Leaf node: If maximum depth is reacehd or # of child nodes are smaller than maximum number of actions
    def get_next_leaf_node(self, current_node):
        if(current_node.depth == self.max_depth or len(current_node.children) < current_node.max_action):
            return current_node
        else:
            child_node = self.get_next_child(current_node)
            return self.get_next_leaf_node(child_node)

    def action_sampler(self, node, num_new_action):
        # Get action values for new actions based on current current node's state
        # Action values are angles. (Assume that radius is constant)
        # Return list of goal positions 
        cur_num_action = len(node.action)
        pose = node.pose 
        max_action = node.max_action
        action_list = np.linspace(-1.0*np.pi, np.pi, num=max_action)
        goal_list = [] 
        for action in action_list[cur_num_action:cur_num_action+num_new_action]:
            goal = np.array([pose[0]+self.horizon_length*math.cos(action), pose[1]+self.horizon_length*math.sin(action)])
            goal_list.append(goal)
        return goal_list, action_list 

    def select_action(self, parent):
        # Select next action for simulation 
        # 1) Expand the leaf node  or 
        # 2) Select the best action w.r.t. UCT criteria 
        
        if(parent.depth == self.max_depth):
            return parent, False 

        leaf_node = self.get_next_leaf_node(parent)

        if(leaf_node.depth <= self.depth_conti):
            update = True
        else:
            update = False 
        if( len(leaf_node.action) < leaf_node.max_action):
            #New action is added 
            new_goal, new_action = self.action_sampler(leaf_node, 1)
            actions, dense_paths = self.path_generator.get_path_set_w_goals(parent.pose, goal_vec)
            #TODO: Add projection with respect to SFC convex
            free_actions, free_dense_paths = self.collision_check(actions, dense_paths)
            new_node = Node(pose = parent.pose, 
                            parent = parent, 
                            name = parent.name + '_action' + str(i), 
                            action = free_actions[action], 
                            dense_path = free_dense_paths[action],
                            is_cont= (leaf_node.depth+1 <=self.depth_conti))
            alpha = 3.0 / (10.0 * (self.max_depth - new_node.depth) - 3.0)
            new_node.set_max_action(self.max_action, alpha)
            leaf_node.add_children(new_node)
            leaf_node.action.append(new_action) 

            return leaf_node.children[len[leaf_node.children]-1], update 
        else:
            #UCT criteria 
            action_node = self.get_next_child(leaf_node)
            return action_node, update 


    def build_initial_action(self, parent):

        self.path_generator.fs = 3 #Number of initial action 
        actions, dense_paths = self.path_generator.get_path_set(parent.pose)

        free_actions, free_dense_paths = self.collision_check(actions, dense_paths)

        for i, action in enumerate(free_actions.keys()):
            # print "Action:", parent.name + '_action' + str(i)
            if len(free_actions[action])!=0:
                # print free_actions[action]
                parent.add_children(Node(pose = parent.pose, 
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
    

    '''
    Make sure update position does not go out of the ranges
    Change best sequence into gradient-updated position 
    '''
    def update_action(self, cur_node, next_node, grad_value):
        # grad_val = best_sequence[-1][:]
        for i, node in enumerate(cur_node.children):
            if(node.pose == next_node.pose):
                selected_idx = i 
                selected_node = cur_node.children[i]
        cur_action = math.atan2(selected_node.pose[1]-cur_node.pose[1], selected_node.pose[0] - cur_node.pose[0])

        step_size = self.grad_step
        update_action = cur_action + step_size * grad_value 

        #Clipping action update into some given bound 
        if( math.abs(update_action - cur_action) > self.update_bound):
            update_action = cur_action + (update_action - cur_action)/math.abs(update_action - cur_action) * self.update_bound 

        new_pose = cur_node.pose + self.horizon_length *np.array([math.cos(update_action), math.sin(update_action)])        
        #Action update 
        cur_node.action[i] = update_action 
        cur_node.children[i].pose = new_pose 
        
        self.update_subtree(cur_node.action[i])

    '''
    Value gradients of action by finite difference.  
    '''
    def get_value_grad(self, cur_node, next_node, action_seq): #best_seq: tuple (path sequence, path cost, reward, number of queries(called))

        cur_pose = cur_node.pose
        next_pose = next_node.pose 
        cur_action = math.atan2(next_pose[1]-cur_pose[1], next_pose[0] - cur_pose[0])
        
        eps = 0.05
        perturb_action1 = cur_action + eps 
        perturb_action2 = cur_action - eps 

        cur_action_list = [cur_action, action_seq]
        perturb_action_list1 = [perturb_action1, action_seq]
        perturb_action_list2 = [perturb_action2, action_seq]

        cur_meas_list = np.empty(shape=(0,2))
        perturb_meas_list1 = np.empty(shape=(0,2))
        perturb_meas_list2 = np.empty(shape=(0,2))
        
        pose = cur_pose 
        for action in cur_action_list:
            pose = pose + np.array([self.horizon_length*math.cos(action), self.horizon_length*math.sin(action_seq)])
            cur_meas_list = np.vstack([cur_meas_list, pose])

        pose = cur_pose 
        for action in perturb_action_list1:
            pose = pose + np.array([self.horizon_length*math.cos(action), self.horizon_length*math.sin(action_seq)])
            perturb_meas_list1 = np.vstack([perturb_meas_list1, pose])

        pose = cur_pose
        for action in perturb_action_list2:
            pose = pose + np.array([self.horizon_length*math.cos(action), self.horizon_length*math.sin(action_seq)])
            perturb_meas_list2 = np.vstack([perturb_meas_list2, pose])
        
        cur_reward = self.get_aq_reward(cur_meas_list, self.belief)
        perturb_reward1 = self.get_aq_reward(perturb_meas_list1, self.belief)
        perturb_reward2 = self.get_aq_reward(perturb_meas_list2, self.belief)

        grad1 = (perturb_reward1 - cur_reward) / eps 
        grad2 = (perturb_reward2 - cur_reward) / (-eps)

        return (grad1+grad2)/2.0 

    def get_aq_reward(self, xobs, belief):
        if self.f_rew == 'mes' or self.f_rew == 'maxs-mes':
            reward = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
        elif self.f_rew == 'exp_improve':
            reward = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
        elif self.f_rew == 'naive':
            # param = sample_max_vals(belief, t=self.t, nK=int(self.param[0]))
            reward = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)#(param, self.param[1]))
        elif self.f_rew == 'naive_value':
            reward = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief, param = self.param)
        else:
            reward = self.acquisition_function(time = self.time, xvals = xobs, robot_model = belief)
        return reward 

    #Update subtree of current node to make sure that actions & states are consistent
    def update_subtree(self, node):
        if(node.depth == self.max_depth or node.children is None):
            return
        else:
            cur_pose = node.pose 
            for idx, action in enumerate(node.action):
                node.children[idx].pose = cur_pose + self.horizon_length * np.array([math.cos(action), math.sin(action)])
                self.update_subtree(node.children[idx])

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

    def visualize_tree(self):
        ranges = (0.0, 20.0, 0.0, 20.0)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(ranges[0:2])
        ax.set_ylim(ranges[2:])
        # for key, value in self.tree.items():
        #     cp = value[0][0]
        #     if(type(cp)==tuple):
        #         x = cp[0]
        #         y = cp[1]
        #         plt.plot(x, y,marker='*')
        plt.show()


class conti_MCTS(MCTS):
    '''MCTS for continuous action'''
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
                path_generator, aquisition_function, f_rew, time, grad_step, lidar, SFC):
        # Call the constructor of the super class
        super(conti_MCTS, self).__init__(ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
                                    path_generator, aquisition_function, f_rew, time, grad_step, lidar, SFC)
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
        self.spent = 0
        self.tree = None
        self.c = 0.1 #Parameter for UCT function 
        self.aquisition_function = aquisition_function
        self.t = time
        self.grad_step = grad_step
        self.lidar = lidar
        self.sdf_map = None
        self.SFC = SFC #SFC Bounding box for optimization 

        self.Nodeset = None 
        self.depth_conti = 2 
        

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

    def get_actions(self):
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
        self.tree = Tree(self.ranges, self.obstacle_world, self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, self.t, max_depth = self.max_depth,
                        max_rollout_depth= self.max_rollout_depth, param = param, c = self.c, grad_step=self.grad_step)

        time_start = time.clock()            
        # while we still have time to compute, generate the tree
        i = 0
        while time.clock() - time_start < self.comp_budget:#i < self.comp_budget:
            i += 1
            next_node, action_update = self.tree.select_action(self.tree.root)
            measurements, action_seq = self.tree.rollout(next_node)
            self.tree.backprop(next_node, measurements, 0.0)

            #TODO: This makes only small number of action update... 
            #      Need another criteria for adequate # of updates....
            if action_update:
                # self.tree.action_update()
                grad_vec = self.tree.get_value_grad(self.tree.root, next_node, action_seq)
                self.tree.update_action(self.tree.root, next_node, grad_vec)
            # gp = copy.copy(self.GP)
            # self.tree.get_next_leaf(gp)

            # if True:
            #     gp = copy.copy(self.GP)
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

