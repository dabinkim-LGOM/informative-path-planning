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
# from MCTS import MCTS 
# from global_BO import ParticleSwarmOpt 


class Node(object):
    def __init__(self, pose, parent, name, action = None, dense_path = None, is_cont = False):
        #is_cont : is this node is a element of continuous-action node set?
        self.pose = pose
        self.name = name
        self.reward = 0.0
        self.nqueries = 0

        self.parent = parent 
        self.children = []
        self.fully_explored = False 
        self.max_action = 8 # Default number 
        #Init action is the first action that child node is added
        #This is for clipping of gradient update  
        if action is None:
            self.action = []
            self.init_action = [] 
        else:
            self.action = action
            self.init_action = action 

        if dense_path is None:
            self.dense_path = []
        else:
            self.dense_path = dense_path
        # self.action = action    
        self.is_cont = is_cont # is cont = true: No dpw, false: dpw

        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth +1 

    def set_max_action(self, max_action, alpha):
        if self.is_cont:
            self.max_action = max_action
        else:
            self.max_action = max_action
            # self.max_action = math.floor(math.log(max_action,alpha)) # Based on progressive widening 

    def add_children(self, child_node):
        if self.children is None:
            self.children = [] 
        self.children.append(child_node)
    
    def remove_children(self, child_node):
        if child_node in self.children: self.children.remove(child_node)
    
    def print_self(self):
        print(self.name)


class Tree(object):
    def __init__(self, ranges, obstacle_world, f_rew, f_aqu,  belief, pose, path_generator, time, max_depth, max_rollout_depth, max_action, param, horizon_length, frontier_size, c, depth_conti, grad_step):
        self.ranges = ranges 
        self.path_generator = path_generator
        self.obstacle_world = obstacle_world
        self.max_depth = max_depth #Maximum tree depth?
        self.max_rollout_depth = max_rollout_depth
        self.max_action = max_action
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
        self.root = Node(pose, parent = None, name = 'root', action = None, dense_path = None, is_cont=True)  

        self.num_action = frontier_size # Number of actions 
        self.c = c 
        self.grad_step = grad_step
        self.update_bound = 0.2

        self.check1 = 0
        self.check2 = 0
        self.check3 = 0

        self.action_check1 = 0
        self.action_check2 = 0
        self.action_check1_1 = 0
        self.action_check1_2 = 0
        self.action_check1_3 = 0 
        
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
        if xobs is None:
            reward += 0.0
            # Backpropagation is ongoing 
            # self.check1+=1
            # return 
        elif len(xobs)==0:
            # self.check2+=1
            return 
        else: 
            self.check3+=1
            reward += self.get_aq_reward(xobs=xobs, belief=self.belief)

        if leaf_node.parent is None:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            # print "Calling backprop on:",
            # leaf_node.print_self()
            # print "nqueries:", leaf_node.nqueries, "reward:", leaf_node.reward / (leaf_node.nqueries+0.01)
            self.check1+=1
            return
        else:
            leaf_node.nqueries += 1
            leaf_node.reward += reward
            # print "Calling backprop on:",
            # leaf_node.print_self()
            # print "nqueries:", leaf_node.nqueries, "reward:", leaf_node.reward
            # print "Parent is ", 
            # leaf_node.parent.print_self()
            self.check2+=1
            self.backprop(leaf_node.parent, None, reward)
            
        
    
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
        action_seq = []
        while cur_depth <= tmp_depth + self.max_rollout_depth:
            #TODO: Modify it to angle action
            # print("cur depth", cur_depth)
            goal_list, action_list = self.action_sampler(current_node, self.max_action, rollout=True)
            # print("Rollout Goal Num: ", len(action_list))
            if(len(action_list) ==0):
                break
            selected_idx = random.choice(range(len(action_list)))
            selected_goal = goal_list[selected_idx]
            selected_action = action_list[selected_idx]
            # print("Cur Pose: ", cur_pose)
            # print("selected goal: ", selected_goal, selected_goal[0], selected_goal[1], cur_pose[2])
            paths, dense_paths = self.path_generator.get_path_set_w_goals(cur_pose, [[selected_goal[0], selected_goal[1], cur_pose[2]]])
            
            action_seq.append(selected_action)
            # if(len(selected_action) == 0):
            #     return cumul_reward
            # print(selected_action)
            # print("Path", paths)
            if(len(paths)==0 or len(paths[0])==0):
                break

            cur_pose = paths[0][-1]
            # data = np.array(paths.get(0))
            # xlocs = np.empty(shape=(0,2))
            # for row in range(data.shape[0]):
            #     pt = np.array([data[row,0], data[row,1]])
            #     if(self.is_bound(pt)):
            #         xlocs = np.vstack([xlocs, pt])
            # print(xlocs)
            
            # print(data)
            # x1 = data[:,0]
            # x2 = data[:,1]
            # xlocs = np.vstack([x1, x2]).T

            #IF rollout node is in map range, add to measurement 
            if(self.is_bound(cur_pose)):
                obs = np.array(cur_pose)
                xlocs = np.vstack([obs[0], obs[1]]).T
            cumul_measruement = np.vstack([cumul_measruement, xlocs])

            current_node = Node(pose = cur_pose, parent = current_node, name = 'Rollout' +str(cur_depth), 
                                action= None, dense_path = None, is_cont=False)            
            cur_depth += 1
        return cumul_measruement, action_seq


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
        # print("Number of  children: ", len(current_node.children))
        # print("Keys: ", vals.keys())
        # print("Values: ", vals.values())
        if(len(current_node.children)==0):
            return -1
        return random.choice([key for key in vals.keys() if vals[key] == max(vals.values())])
    
    #Find leaf node based on UCT criteria
    #Leaf node: If maximum depth is reacehd or # of child nodes are smaller than maximum number of actions
    def get_next_leaf_node(self, current_node):
        # flag = -1
        if( len(current_node.children) < current_node.max_action and (not current_node.fully_explored) ):
            return current_node, 0
        elif(current_node.max_action ==0):
            current_node.fully_explored = True 
            print("#### Maximum Action is Zero. Current node is selected ####")
            return current_node, 1
        else:
            if(current_node.depth == self.max_depth):
                return current_node, 2
            else:
                child_node = self.get_next_child(current_node)
                if(child_node ==-1):
                    return current_node, 3 
            return self.get_next_leaf_node(child_node)

    def action_sampler(self, node, num_new_action, rollout=False):
        # Get action values for new actions based on current current node's state
        # Action values are angles. (Assume that radius is constant)
        # Return list of goal positions 
        cur_num_action = len(node.action)
        pose = node.pose 
        max_action = node.max_action
        action_list = np.linspace(-1.0*np.pi, np.pi, num = max_action)
        goal_list = []
        selected_action_list = []
        num_select = 0  
        idx = 0
        
        if rollout:
            num_new_action = node.max_action
            # selected_action_list = action_list 
            for action in action_list:
                goal = np.empty(shape=2)
                # goal[0] = pose[0][0] + self.horizon_length*math.cos(action)
                # goal[1] = pose[1][0] + self.horizon_length*math.sin(action) 
                goal = pose[0:1][0] + self.horizon_length*np.array([math.cos(action), math.sin(action)])
                
                # print("pose: ", pose[0:1][0], "hl", self.horizon_length, "action: " , np.array([math.cos(action), math.sin(action)]))
                # print("goal: ", goal)
                if(self.is_bound(goal)):
                    goal_list.append(goal)
                    selected_action_list.append(action)
        else:
            while(num_select < num_new_action):
                if(cur_num_action+idx > len(action_list)-1):
                    # print("[WARNING] Number of action exceeds")
                    break
                action = action_list[cur_num_action+idx]
                goal = np.empty(shape=(2))
                goal[0] = pose[0] + self.horizon_length*math.cos(action)
                goal[1] = pose[1] + self.horizon_length*math.sin(action)
                # goal = pose[0:1] + self.horizon_length*np.array([math.cos(action), math.sin(action)])
                if(self.is_bound(goal)):
                    goal_list.append(goal)
                    selected_action_list.append(action)
                    num_select +=1
                idx +=1

            # for action in action_list[cur_num_action:cur_num_action+num_new_action]:
            #     goal = np.array([pose[0]+self.horizon_length*math.cos(action), pose[1]+self.horizon_length*math.sin(action)])
            #     goal_list.append(goal)
        return goal_list, selected_action_list 
    
    def is_bound(self, point):
        if point[0] < self.ranges[0] or point[0] > self.ranges[1] or point[1] < self.ranges[2] or point[1] > self.ranges[3]:
            return False
        else:
            return True 

    def select_action(self, parent):
        # Select next action for simulation 
        # 1) Expand the leaf node  or 
        # 2) Select the best action w.r.t. UCT criteria 
        
        if(parent.depth == self.max_depth):
            return parent, False 

        leaf_node, flag = self.get_next_leaf_node(parent)
        # print(leaf_node)
        # print(flag)
        if(leaf_node.name == 'root'):
            print("[SELECTION] is ROOT NODE")
            if flag==0:
                print("[SELECTION] MAX CHILDREN IS NOT REACHED")
            elif flag==1:
                print("[SELECTION] MAX ACTION IS ZERO")
            elif flag==2:
                print("[SELECTION] MAX DEPTH IS REACHED")
            elif flag==3:
                print("[SELECTION] CHILD NODE IS -1")
            else:
                print("[SELECTION] NO OPTION")
        # print("[SELECT NODE]: Leaf node is selected")
        # leaf_node.print_self()

        # print("Number of children of leaf node: ", len(leaf_node.children))
        # Leaf node is continuous-action node
        if(leaf_node.depth <= self.depth_conti):
            update = True
        else:
            update = False 

        # Determine progressive widening criteria which determines whether expand action space 
        alpha = 3.0 / (10.0 * (self.max_depth - leaf_node.depth) - 3.0)
        if leaf_node.is_cont:
            dpw = True
        else:
            if math.floor(leaf_node.nqueries**alpha) >= len(leaf_node.children):
                dpw = True
            else:
                dpw = False
        
        if(len(leaf_node.action) >= leaf_node.max_action):
            # print "[LENGTH EXCEEDS]",
            # leaf_node.print_self()
            # print("CUR_ACTION_LEN: ", len(leaf_node.action), " MAXIMUM ACTION: ", leaf_node.max_action)
            # print("CUR_CHILD_LEN: ", len(leaf_node.children))
            # if(leaf_node.fully_explored):
            #     print("FULLY EXPLORED!")
            # else:
            #     print("NOT FULLY EXPLORED!")
            # # print("CHILD_ACT_LEN: ", max( len(child.action) for child in leaf_node.children) )
            # print("NODE DEPTH: ", leaf_node.depth, " MAX DEPTH: ", self.max_depth)
            # # len(leaf_node.children[0].action))
            self.action_check1_1 +=1 

        elif(leaf_node.fully_explored):
            self.action_check1_2 +=1
        elif(not dpw):
            self.action_check1_3 +=1

        if( len(leaf_node.action) < leaf_node.max_action and not leaf_node.fully_explored and dpw):
            self.action_check1 += 1
            #New action is added 
            new_goal, new_action = self.action_sampler(leaf_node, 1)
            # print(len(new_goal))
            if(len(new_goal)==0):
                leaf_node.fully_explored = True
                print("Fully Explored")
                return leaf_node, False
            # print("New goal", new_goal[0])

            if not self.is_bound(new_goal[0]):
                return parent, False 

            actions, dense_paths = self.path_generator.get_path_set_w_goals(parent.pose, new_goal)
            #TODO: Add projection with respect to SFC convex
            free_actions, free_dense_paths = self.collision_check(actions, dense_paths)
            # print("Free action", free_actions)
            new_path = free_actions.get(0)
            new_dense_path = free_dense_paths.get(0)

            #Generate New Node 
            new_node = Node(pose = new_path[-1], 
                            parent = leaf_node, 
                            name = leaf_node.name + '_action' + str(len(leaf_node.children)+1), 
                            action = None, 
                            dense_path = None,
                            is_cont= (leaf_node.depth+1 <=self.depth_conti))
            # new_node.print_self()
            alpha = 3.0 / (10.0 * (self.max_depth - new_node.depth) - 3.0)

            new_node.set_max_action(self.max_action, alpha)
            #Update action / dense path to parent node             
            #Add children to parent node 
            leaf_node.add_children(new_node)
            leaf_node.action.append(new_action) 
            leaf_node.init_action.append(new_action)
            leaf_node.dense_path.append(new_dense_path) 

            return new_node, new_node.is_cont
        else:
            self.action_check2 += 1
            #UCT criteria 
            if(leaf_node.depth == self.max_depth):
                return leaf_node, update
            else:
                action_node = self.get_next_child(leaf_node)
                if(action_node == -1):
                    return leaf_node, update
                # print("In UCT criteria", action_node.name)
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
                                        action = None, 
                                        dense_path = None,
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
        '''
        cur_node: parent node 
        next_node: child node which wants to be updated 
        '''
        # grad_val = best_sequence[-1][:]
        try:
            if(len(cur_node.children)==0):
                raise Exception('[GRAD UPDATE] No child for current node')
        except Exception as e:
            print('Exception occurs at UPDATE ACTION function', e)
        
        #Find index of next_node among cur_node's childrens list
        selected_idx = 0
        selected_node = cur_node.children[0]
        find = False 
        for i, node in enumerate(cur_node.children):
            # print("node pose: ", node.pose, "next node pose", next_node.pose)
            if(node.pose[0] == next_node.pose[0] and node.pose[1] == next_node.pose[1]):
                selected_idx = i 
                selected_node = cur_node.children[i]
                find = True 
        if(find==False):
            print("[UPDATE ACTION]: Index Not Found Error!")

        # cur_action = math.atan2(selected_node.pose[1]-cur_node.pose[1], selected_node.pose[0] - cur_node.pose[0])
        cur_action = math.atan2(next_node.pose[1]-cur_node.pose[1], next_node.pose[0] - cur_node.pose[0])
        init_action = cur_node.init_action[selected_idx]

        step_size = self.grad_step
        update_action = cur_action + step_size * grad_value 

        #Clipping action update into some given bound 
        if( abs(update_action - init_action) > self.update_bound):
            update_action = init_action + (update_action - init_action)/abs(update_action - init_action) * self.update_bound 
        
        # new_xy = cur_node.pose[0:1] + self.horizon_length *np.array([math.cos(update_action), math.sin(update_action)])
        new_xy = np.empty(shape=2)
        new_xy[0] = cur_node.pose[0] + self.horizon_length*math.cos(update_action)
        new_xy[1] = cur_node.pose[1] + self.horizon_length*math.sin(update_action)
        new_pose = np.array([new_xy[0], new_xy[1], update_action])
        #Action update 
        cur_node.action[selected_idx] = update_action 
        cur_node.children[selected_idx].pose = new_pose 
        # print('#############Updated Node##############')
        # cur_node.children[i].print_self()
        # print('# of Action: ', len(cur_node.children[i].action))
        # print('# of Children: ', len(cur_node.children[i].children))
        self.update_subtree(cur_node.children[selected_idx])

    '''
    Value gradients of action by finite difference.  
    '''
    def get_value_grad(self, cur_node, next_node, action_seq): #best_seq: tuple (path sequence, path cost, reward, number of queries(called))

        cur_pose = cur_node.pose
        next_pose = next_node.pose 
        cur_action = math.atan2(next_pose[1]-cur_pose[1], next_pose[0] - cur_pose[0])
        
        #Action sequences with or without perturbations 
        eps = 0.05
        perturb_action1 = cur_action + eps
        perturb_action2 = cur_action - eps

        cur_action_list = action_seq
        cur_action_list.insert(0, cur_action)

        perturb_action_list1 = action_seq 
        perturb_action_list1.insert(0, perturb_action1)

        perturb_action_list2 = action_seq 
        perturb_action_list2.insert(0, perturb_action2)
        
        # Measurement positions according to respective action lists 
        cur_meas_list = np.empty(shape=(0,2))
        perturb_meas_list1 = np.empty(shape=(0,2))
        perturb_meas_list2 = np.empty(shape=(0,2))
        
        pose = cur_pose[0:1]
        for action in cur_action_list:
            # pose[0] = pose[0] + self.horizon_length*math.cos(action)
            # pose[1] = pose[1] + self.horizon_length*math.sin(action)
            pose = pose + self.horizon_length*np.array([math.cos(action), math.sin(action)])
            cur_meas_list = np.vstack([cur_meas_list, pose])

        pose = cur_pose[0:1]
        for action in perturb_action_list1:
            # pose[0] = pose[0] + self.horizon_length*math.cos(action)
            # pose[1] = pose[1] + self.horizon_length*math.sin(action)
            pose = pose + self.horizon_length*np.array([math.cos(action), math.sin(action)])
            perturb_meas_list1 = np.vstack([perturb_meas_list1, pose])

        pose = cur_pose[0:1]
        for action in perturb_action_list2:
            # pose[0] = pose[0] + self.horizon_length*math.cos(action)
            # pose[1] = pose[1] + self.horizon_length*math.sin(action)
            pose = pose + self.horizon_length*np.array([math.cos(action), math.sin(action)])
            perturb_meas_list2 = np.vstack([perturb_meas_list2, pose])
        
        # Reward computation 
        cur_reward = self.get_aq_reward(cur_meas_list, self.belief)
        perturb_reward1 = self.get_aq_reward(perturb_meas_list1, self.belief)
        perturb_reward2 = self.get_aq_reward(perturb_meas_list2, self.belief)

        # Gradient computation 
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
        if(node.depth == self.max_depth or len(node.children)==0):
            return
        else:
            cur_pose = node.pose 
            # print('Current Node Name: ')
            # node.print_self()
            if len(node.action) >0:
                for idx, action in enumerate(node.action):
                    if(type(action) is not float):
                        action = action[0]
                    # print('children numb: ', len(node.children), 'action num: ', len(node.action))
                    # print('action: ', action)
                    # print('children : ', node.children[idx].pose)
                    node.children[idx].pose = np.array([cur_pose[0] + self.horizon_length * math.cos(action),
                                                        cur_pose[1] + self.horizon_length * math.sin(action), action])
                    # node.children[idx].pose[2] = action 
                    self.update_subtree(node.children[idx])

    def print_tree_structure(self):
        self.print_id(self.root) 

    def print_id(self, cur_node):
        if cur_node.children is None:
            cur_node.print_self()
            return
        else:
            for child in cur_node.children:
                child.print_self()
                self.print_id(child)

    def print_tree(self):
        counter = self.print_helper(self.root)
        print("# nodes in tree:", counter)

    def print_helper(self, cur_node):
        if cur_node.children is None or len(cur_node.children)==0:
            # cur_node.print_self()
            # print cur_node.name
            return 1
        else:
            # cur_node.print_self()
            # print "\n"
            counter = 0 
            # print("Number of child: ", len(cur_node.children))
            for child in cur_node.children:
                counter += self.print_helper(child)
            # print("counter: ", counter)
            return counter

    def position_span_tree(self, cur_node):
        if cur_node.children is None or len(cur_node.children)==0:
            return cur_node.pose
        else:
            pos_vec = np.empty(shape=(0,3))
            for child in cur_node.children:
                # print("Pos vec", pos_vec)
                # print("Child", self.position_tree(child))
                pos_vec = np.vstack([pos_vec, self.position_span_tree(child)])
            return pos_vec
    
    def position_first_depth(self, cur_node):
        pos_vec = np.empty(shape=(0,3))
        for child in cur_node.children:
            pos_vec = np.vstack([pos_vec, child.pose])
        return pos_vec

    def position_init_depth(self, cur_node):
        pos_vec = np.empty(shape=(0,2))
        for action in cur_node.init_action:
            # print("Action", cur_node.init_action)
            # print("Size of pose", cur_node.pose[1])
            
            action = action[0]
            init_pose = np.empty(shape=(2,1))
            init_pose[0] = cur_node.pose[0] + self.horizon_length*math.cos(action)
            init_pose[1] = cur_node.pose[1] + self.horizon_length*math.sin(action)
            
            # init_pose = np.array(cur_node.pose[0:1]) + self.horizon_length*np.array([math.cos(action), math.sin(action)])
            # print("Action", action, "Init pose", init_pose)
            
            pos_vec = np.vstack([pos_vec, init_pose.T])
        return pos_vec

    def visualize_tree(self):
        cur_pos = self.root.pose 
        ranges = (cur_pos[0]-20, cur_pos[0]+20, cur_pos[1]-20, cur_pos[1]+20)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(ranges[0:2])
        ax.set_ylim(ranges[2:])
        # pos_vec = self.position_span_tree(self.root) # All nodes in tree 
        pos_vec = self.position_first_depth(self.root)
        for pos in pos_vec:
            x = pos[0]
            y = pos[1]
            plt.plot(x,y, marker='*', c='b')       
        plt.plot(cur_pos[0], cur_pos[1], marker='x', c='r')
        # for key, value in self.tree.items():
        #     cp = value[0][0]
        #     if(type(cp)==tuple):
        #         x = cp[0]
        #         y = cp[1]
        #         plt.plot(x, y,marker='*')
        plt.show()

    def get_action_value(self, prev_node, next_node):
        cur_action = math.atan2(next_node.pose[1]-prev_node.pose[1], next_node.pose[0] - prev_node.pose[0])
        return cur_action


class conti_MCTS(object):
    '''MCTS for continuous action'''
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
                path_generator, aquisition_function, f_rew, horizon_length, time, grad_step, lidar, SFC):
        # Call the constructor of the super class
        # super(conti_MCTS, self).__init__(ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
        #         path_generator, aquisition_function, f_rew, time, grad_step, lidar, SFC)
        # self.tree_type = 'dpw'
        self.ranges = ranges
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
        self.horizon_length = horizon_length
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
        self.depth_conti = 1
        
        # The differnt constatns use logarthmic vs polynomical exploriation
        if self.f_rew == 'mean':
            # if self.tree_type == 'belief':
            self.c = 0.1
            # self.c = 1000
            # elif self.tree_type == 'dpw':
            # self.c = 5000
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
        print('c value: ', self.c)

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

        max_action = 8
        # initialize tree
        self.tree = Tree(self.ranges, self.obstacle_world, self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, self.t, 
                        max_depth = self.max_depth, max_rollout_depth= self.max_rollout_depth, max_action = max_action, param = param, horizon_length= self.horizon_length, frontier_size=self.frontier_size,
                        c = self.c, depth_conti=2, grad_step=self.grad_step)

        time_start = time.clock()
        # while we still have time to compute, generate the tree
        i = 0
        while time.clock() - time_start < self.comp_budget:#i < self.comp_budget:
            # print("############################CHECK####################################")
            i += 1
            next_node, action_update = self.tree.select_action(self.tree.root)
            measurements, action_seq = self.tree.rollout(next_node)
            self.tree.backprop(next_node, measurements, 0.0)

            # action_update = False 
            # TODO: Value gradient should perform every step 
            #       Find depth-1 node from current node & get action sequence by tracking backward 
            if action_update:
                # self.tree.action_update()
                # print("[BEFORE UPDATE]: ")
                while(next_node.depth>1):
                    # next_node.print_self()
                    prev_node = next_node 
                    next_node = next_node.parent
                    action_tmp = self.tree.get_action_value(prev_node, next_node) 
                    action_seq.insert(0, action_tmp)
                # next_node.print_self()
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
        # self.tree.visualize_tree()

        # self.tree.print_tree_structure()

        print([(node.nqueries, node.reward/(node.nqueries+0.1)) for node in self.tree.root.children])

        # print("Analyze Backpropagation")
        # print("Check 1: ", self.tree.check1, " Check 2: ", self.tree.check2, " Check 3: ", self.tree.check3)
        print("Analyze Select Action")
        print("Make Children: ", self.tree.action_check1, " Select UCT children: ", self.tree.action_check2)
        print("Lenght Exceeds Max: ", self.tree.action_check1_1, " Fully Explored: ", self.tree.action_check1_2, " DPW: ", self.tree.action_check1_3)
        #Executing the first action 
        
        # best_child = self.tree.root.children[np.argmax([node.nqueries for node in self.tree.root.children])]
        best_child = self.tree.root.children[np.argmax([node.reward/node.nqueries for node in self.tree.root.children])]
        # best_child = random.choice([node for node in self.tree.root.children if node.nqueries == max([n.nqueries for n in self.tree.root.children])])
        all_vals = {}
        for i, child in enumerate(self.tree.root.children):
            all_vals[i] = child.reward / (float(child.nqueries)+0.1)
            # print(str(i) + " is " + str(all_vals[i]))

        paths, dense_paths = self.path_generator.get_path_set(self.cp)

        self.path_generator.cp = self.cp 
        best_path, best_dense_path = self.path_generator.make_sample_paths_w_goals([best_child.pose])        
        # return best_child.action, best_child.dense_path, best_child.reward/(float(best_child.nqueries)+1.0), paths, all_vals, self.max_locs, self.max_val, self.target

        return best_path.get(0), best_dense_path.get(0), best_child.reward/(float(best_child.nqueries)+1.0) 


        # get the best action to take with most promising futures, base best on whether to
        # consider cost
        #best_sequence, best_val, all_vals = self.get_best_child()

        #Document the information
        #print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        #logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))
        #np.save('./figures/' + self.f_rew + '/tree_' + str(t) + '.npy', self.tree)
        #return self.tree[best_sequence][0], self.tree[best_sequence][1], best_val, paths, all_vals, self.max_locs, self.max_val

