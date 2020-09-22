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


class MCTS(object):
    '''Class that establishes a MCTS for nonmyopic planning'''
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, max_depth, max_rollout_depth, frontier_size,
                path_generator, aquisition_function, f_rew, time, gradient_on, grad_step, lidar, SFC):
        '''Initialize with constraints for the planning, including whether there is 
           a budget or planning horizon
           budget - length, time, etc to consider
           belief - GP model of the robot current belief state
           initial_pose - (x,y,rho) for vehicle'''
        self.ranges = ranges
        self.budget = computation_budget
        self.GP = belief
        self.cp = initial_pose
        self.limit = max_rollout_depth
        self.frontier_size = frontier_size
        self.path_generator = path_generator
        self.obstacle_world = obstacle_world
        # self.default_path_generator = Path_Generator(frontier_size, )
        self.spent = 0
        self.tree = None
        self.c = 0.1 #Parameter for UCT function 
        self.aquisition_function = aquisition_function
        self.t = time
        self.gradient_on = gradient_on
        self.grad_step = grad_step
        self.lidar = lidar
        self.sdf_map = None
        self.SFC = SFC #SFC Bounding box for optimization 

    def get_actions(self):

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

    '''
    Make sure update position does not go out of the ranges
    Change best sequence into gradient-updated position 
    '''
    def update_action(self,best_sequence):
        grad_val = best_sequence[-1][:]
        grad_x = grad_val[0]
        grad_y = grad_val[1]

        # step_size = 0.05
        step_size = self.grad_step
        last_action = best_sequence[0][-1][:]

        last_action_x = last_action[0] + step_size * grad_x
        last_action_y = last_action[1] + step_size * grad_y

        #Restrict updated action inside the environment
        if(last_action_x < self.ranges[0]):
            last_action_x = self.ranges[0] + 0.1
        elif(last_action_x > self.ranges[1]):
            last_action_x = self.ranges[1] - 0.1
        
        if(last_action_y < self.ranges[2]):
            last_action_y = self.ranges[2] + 0.1
        elif(last_action_y > self.ranges[3]):
            last_action_y = self.ranges[3] - 0.1

        last_action_update = (last_action_x, last_action_y, last_action[2])

        best_sequence[0][-1] = last_action_update

        return best_sequence


    '''
    Value gradients of action by finite difference.  
    '''
    def get_value_grad(self, cur_node, cur_seq, cur_reward): #best_seq: tuple (path sequence, path cost, reward, number of queries(called))
        
        # init_node = self.tree[cur_seq[0][:]]
        path_seq = []
        for seq in cur_seq:
            for tmp_path_seq in self.tree[seq][0][:]:
                path_seq.append(tmp_path_seq)
        # cur_node_reward = init_node[2]
        # num_queri = init_node[3]
        
        if(len(path_seq)>=2):
            final_action = path_seq[-1]
            new_x_seq = path_seq[:]
            new_y_seq = path_seq[:]

            x = final_action[0]
            y = final_action[1]
            yaw = final_action[2]
            eps = 0.1
            step = 1.0
            x_dif = x + eps * step 
            y_dif = y + eps * step

            new_x_action = (x_dif, y, yaw)
            new_y_action = (x, y_dif, yaw)

            new_x_seq[-1] = new_x_action
            new_y_seq[-1] = new_y_action
            reward_x = self.get_reward(new_x_seq)
            reward_y = self.get_reward(new_y_seq)

            grad_x = (reward_x- cur_reward)/ eps
            grad_y = (reward_y - cur_reward) / eps
        return [grad_x, grad_y]

    def generate_sdfmap(self, cp):
        # x, y = self.cp[0], self.cp[1]
        try:
            length = np.array([15, 15])
            sdf_map = sdflib.sdf_map(self.ranges, self.obstacle_world, self.lidar, cp, length)
            return sdf_map
        except Exception as ex:
            print("During Generating SDFMAP", ex)

    def update_sdfmap(self, node):
        if(self.sdf_map is not None):
            # print(type(node))
            print(node)

    def visualize_tree(self):
        ranges = (0.0, 20.0, 0.0, 20.0)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(ranges[0:2])
        ax.set_ylim(ranges[2:])
        for key, value in self.tree.items():
            cp = value[0][0]
            if(type(cp)==tuple):
                x = cp[0]
                y = cp[1]
                plt.plot(x, y,marker='*')
        plt.show()

    def collision_check(self, path_dict):
        free_paths = {}
        for key,path in path_dict.items():
            is_collision = 0
            for pt in path:
                if(self.obstacle_world.in_obstacle(pt, 3.0)):
                    is_collision = 1
            if(is_collision == 0):
                free_paths[key] = path
        
        return free_paths 
    
    def initialize_tree(self):
        '''Creates a tree instance, which is a dictionary, that keeps track of the nodes in the world'''
        tree = {}
        #(pose, number of queries)
        tree['root'] = (self.cp, 0)
        actions, _ = self.path_generator.get_path_set(self.cp)
        feas_actions = self.collision_check(actions)

        for action, samples in feas_actions.items():
            #(samples, cost, reward, number of times queried)
            cost = np.sqrt((self.cp[0]-samples[-1][0])**2 + (self.cp[1]-samples[-1][1])**2)
            tree['child '+ str(action)] = (samples, cost, 0, 0)
        return tree

    def tree_policy(self):
        '''Implements the UCB policy to select the child to expand and forward simulate'''
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

    def rollout_policy(self, node, budget):
        '''Select random actions to expand the child node'''
        sequence = [node] #include the child node
        #TODO use the cost metric to signal action termination, for now using horizon
        for i in xrange(self.limit):
            actions, _ = self.path_generator.get_path_set(self.tree[node][0][-1]) #plan from the last point in the sample
            # feas_actions = self.collision_check(actions)
            a = np.random.randint(0,len(actions)) #choose a random path
            # while(not (a in feas_actions)):
            #     a = np.random.randint(0,len(actions)) #choose a random path
            
            
            #TODO add cost metrics
#             best_path = actions[a]
#             if len(best_path) == 1:
#                 best_path = [(best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)]
#             elif best_path[-1][0] < -9.5 or best_path[-1][0] > 9.5:
#                 best_path = (best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)
#             elif best_path[-1][1] < -9.5 or best_path[-1][0] >9.5:s
#                 best_path = (best_path[-1][0],best_path[-1][1],best_path[-1][2]-1.14)
#             else:
#                 best_path = best_path[-1]
            self.tree[node + ' child ' + str(a)] = (actions[a], 0, 0, 0) #add random path to the tree
            node = node + ' child ' + str(a)
            sequence.append(node)
        return sequence #return the sequence of nodes that are made

    def backprop(self, reward, sequence, value_grad):
        '''Propogate the reward for the sequence'''
        #TODO update costs as well
        self.tree['root'] = (self.tree['root'][0], self.tree['root'][1]+1)
        for seq in sequence:
            # value_grad = 0
            if(len(self.tree[seq])>4):
                samples, cost, rew, queries, value_grad = self.tree[seq]
            else:
                samples, cost, rew, queries = self.tree[seq]
            queries += 1
            n = queries
            rew = ((n-1)*rew+reward)/n
            self.tree[seq] = (samples, cost, rew, queries)
            if(value_grad!=None):
                # print("In Here!")
                self.tree[seq] = (samples, cost, rew, queries, value_grad)


    def get_reward(self, sequence):
        '''Evaluate the sequence to get the reward, defined by the percentage of entropy reduction'''
        # The process is iterated until the last node of the rollout sequence is reached 
        # and the total information gain is determined by subtracting the entropies 
        # of the initial and final belief space.
        # reward = infogain / Hinit (joint entropy of current state of the mission)
        sim_world = self.GP
        samples = []
        obs = []
        for seq in sequence:
            if(type(seq)== tuple):
                samples.append([seq])
            else:
                samples.append(self.tree[seq][0])
        obs = list(chain.from_iterable(samples))
        if(self.aquisition_function==aqlib.mves ):
            #TODO: Fix the paramter setting. This leads to wrong MES acquisition function computation.
            # maxes, locs, funcs = sample_max_vals(robot_model=sim_world, t=t, nK = 3, nFeatures = 200, visualize = False, obstacles=obslib.FreeWorld(), f_rew='mes'): 
            return self.aquisition_function(time = self.t, xvals = obs, param= (self.max_val, self.max_locs, self.target), robot_model = sim_world)
        else:
            return self.aquisition_function(time = self.t, xvals = obs, robot_model = sim_world)
    

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




class Node(object):
    def __init__(self, pose, parent, name, action = None, dense_path = None, zvals = None):
        self.pose = pose
        self.name = name
        self.zvals = zvals
        self.reward = 0.0
        self.nqueries = 0
        
        # Parent will be none if the node is a root
        self.parent = parent
        self.children = None

        # Set belief or belief action node
        if action is None:
            self.node_type = 'B'
            self.action = None 
            self.dense_path = None

            # If the root node, depth is 0
            if parent is None:
                self.depth = 0
            else:
                self.depth = parent.depth + 1
        else:
            self.node_type = 'BA'
            self.action = action
            self.dense_path = dense_path
            self.depth = parent.depth

    def add_children(self, child_node):
        if self.children is None:
            self.children = []
        self.children.append(child_node)
    
    def remove_children(self, child_node):
        if child_node in self.children: self.children.remove(child_node)

    def print_self(self):
        print(self.name)

class Tree(object):
    def __init__(self, ranges, obstacle_world, f_rew, f_aqu,  belief, pose, path_generator, t, max_depth, max_rollout_depth, turning_radius, param, c, gradient_on, grad_step):
        self.ranges = ranges 
        self.path_generator = path_generator
        self.obstacle_world = obstacle_world
        self.max_depth = max_depth #Maximum tree depth?
        self.max_rollout_depth = max_rollout_depth
        self.param = param
        self.t = t
        self.f_rew = f_rew
        self.aquisition_function = f_aqu
        self.turning_radius = turning_radius
        self.c = c
        self.gradient_on = gradient_on
        self.grad_step = grad_step
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
    
    def get_next_leaf(self, belief):
        #print "Calling next with root"
        
        next_leaf, reward = self.leaf_helper(self.root, reward = 0.0,  belief = belief, rollout_flag=1) 
        #print "Next leaf:", next_leaf
        #print "Reward:", reward
        self.backprop(next_leaf, reward)
    
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

    #TODO: Rollout & Node expanding policy is not separated. 
    #Rollout flag -> If rollout_flag=1, do 
    def leaf_helper(self, current_node, reward, belief, rollout_flag):
        if current_node.node_type == 'B':
            # Root belief node
            if current_node.depth == self.max_depth:
                #print "Returning leaf node:", current_node.name, "with reward", reward
                if rollout_flag:
                    rollout_reward = self.rollout(current_node, belief)
                    return current_node, reward+rollout_reward
                else:
                    return current_node, reward
            # Intermediate belief node
            else:
                if current_node.children is None:
                    self.build_action_children(current_node)
                    if rollout_flag:
                        rollout_reward = self.rollout(current_node, belief)
                        return current_node, rollout_reward 

                # If no viable actions are avaliable
                if current_node.children is None:
                    return current_node, reward

                child = self.get_next_child(current_node) #Select with UCT rule 
                # print("Selecting next action child:", child.name)

                # Recursive call
                return self.leaf_helper(child, reward, belief, rollout_flag)

        # At random node, after selected action from a specific node
        elif current_node.node_type == 'BA':
            # Copy old belief
            #gp_new = copy.copy(current_node.belief) 
            #gp_new = current_node.belief

            # Sample a new set of observations and form a new belief
            #xobs = current_node.action
            # print "Type", type(current_node.action)
            # print current_node.action
            obs = np.array(current_node.action)
            # print "OBS: ", obs
            if(len(obs)==0):
                print("Observation set is empty", current_node)
                # continue 
            xobs = np.vstack([obs[:,0], obs[:,1]]).T

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

            if current_node.children is not None:
                alpha = 3.0 / (10.0 * (self.max_depth - current_node.depth) - 3.0)
                # print("Cur depth: ", current_node.depth, "Alpha ", alpha)
                nchild = len(current_node.children)
                # print "Current depth:", current_node.depth, "alpha:", alpha
                
                '''
                Progressive Widening
                 : Do not make more child(action), and choose from among current child(action)
                   Check it's number of parent node's children. (PW for action nodes, not observations)     
                '''
                nchild = len(current_node.parent.children)
                # print "First:", np.floor(nchild ** alpha)
                # print "Second:", np.floor((nchild - 1) ** alpha)
                # print "Number of child: ", nchild

                '''
                Gradient Update:  
                    Overall - Get gradient-updated action from current action-node & add to its parent state-node. 
                    1) With Progressive Widening, not too many gradient action-nodes would be generated. (Action state is clipped)
                    2) Given execution horizon, gradient-update is projected toward feasible region (SFC & Radius)
                '''
                #TODO: SFC Selection

                '''If PW condition is satisfied, skip the gradient update step and return it's child node '''
                if current_node.depth < self.max_depth - 1 and np.floor(nchild ** alpha) == np.floor((nchild - 1) ** alpha):
                    # #print "Choosing from among current nodes"
                    # #child = random.choice(current_node.children)
                    # #print "number quieres:", nqueries
                    child = random.choice(current_node.children)
                    nqueries = [node.nqueries for node in current_node.children]
                    #select node which has the minimum queuried number & random if ties 
                    child = random.choice([node for node in current_node.children if node.nqueries == min(nqueries)])
                    return self.leaf_helper(child, reward + r, belief, 0) #No rollout if PW is called 

                    # if True:
                    #     belief.add_data(xobs, child.zvals)
                    #print "Selcted child:", child.nqueries
                    # print current_node.children
                    
                #What to return -> Current node's children 
                elif current_node.depth < self.max_depth - 1:
                    print "##########################Grad Called################################"
                    # grad_val = self.get_value_grad(current_node)
                    actions = current_node.action 
                    dense_paths = current_node.dense_path
                    grad_updated_action = self.update_action(actions)
                    pg = Dubins_EqualPath_Generator(1,1.0, self.turning_radius, 0.5, self.ranges)
                    pg.cp = current_node.pose
                    pg.goals = [grad_updated_action[-1]]
                    # print "Update Action: ", grad_updated_action
                    # print "Goals: ", pg.goals
                    new_action, new_dense_path = pg.make_sample_paths()
                    new_action[0].append(grad_updated_action[-1])
                    new_dense_path[0].append(grad_updated_action[-1])
                    # print "New action: ", new_action
                    # actions[action] = new_action[0]
                    # dense_paths[action] = new_dense_path[0]

                    for i, action in enumerate(new_action.keys()):
                        # print "new action ", new_action
                        grad_update_node = Node(pose=current_node.parent.pose, parent=current_node.parent, 
                                                name=current_node.parent.name + '_action'+'_grad', 
                                                action = new_action[action], dense_path = new_dense_path[action], zvals = None)
                    current_node.parent.add_children(grad_update_node)
                    return self.leaf_helper(current_node.children[0], reward + r, belief, 1) #Yes rollout if PW is not called
                
            '''
            If there is already an child node(state-node), we do not need to add more children. (Previously, for cont. observation, they used)
            '''
            pose_new = current_node.dense_path[-1]
            child = Node(pose = pose_new, 
                        parent = current_node, 
                        name = current_node.name + '_depth' + str(current_node.depth + 1), 
                        action = None, 
                        dense_path = None, 
                        #  zvals = zobs
                        zvals = None
                        )
            # print("Adding next state child:", child.name)
            current_node.add_children(child)

            # Recursive call
            return self.leaf_helper(child, reward + r, belief, rollout_flag)

            # if True:
            #     if belief.model is None:
            #         n_points, input_dim = xobs.shape
            #         zmean, zvar = np.zeros((n_points, )), np.eye(n_points) * belief.variance
            #         zobs = np.random.multivariate_normal(mean = zmean, cov = zvar)
            #         zobs = np.reshape(zobs, (n_points, 1))
            #     else:
            #         zobs = belief.posterior_samples(xobs, full_cov = False, size = 1)
            #         n_points, input_dim = xobs.shape
            #         zobs = np.reshape(zobs, (n_points,1))

            #     # print(xobs)
            #     # print(type(zobs))
            #     belief.add_data(xobs, zobs) # Work as continuous observation 

            # else:
            #     zobs = belief.posterior_samples(xobs, full_cov = False, size = 1)
            #     n_points, input_dim = xobs.shape
            #     zobs = np.reshape(zobs, (n_points,1))

            # belief.add_data(xobs, zobs) #If we add tree's belief the new query points, it means it is continuous-observation case. 

    # def projection(self):
    #   

    '''
    Make sure update position does not go out of the ranges
    Change best sequence into gradient-updated position 
    '''
    def update_action(self, cur_action):
        # grad_val = best_sequence[-1][:]
        grad_val = self.get_value_grad(cur_action)
        grad_x = grad_val[0]
        grad_y = grad_val[1]

        # step_size = 0.05
        step_size = self.grad_step
        last_action = cur_action[-1]
        # print last_action
        last_action_x = last_action[0] + step_size * grad_x
        last_action_y = last_action[1] + step_size * grad_y
        print "Grad value: ", grad_x, " ", grad_y
        #Restrict updated action inside the environment
        if(last_action_x < self.ranges[0]):
            last_action_x = self.ranges[0] + 0.1
        elif(last_action_x > self.ranges[1]):
            last_action_x = self.ranges[1] - 0.1
        
        if(last_action_y < self.ranges[2]):
            last_action_y = self.ranges[2] + 0.1
        elif(last_action_y > self.ranges[3]):
            last_action_y = self.ranges[3] - 0.1

        last_action_update = (last_action_x, last_action_y, last_action[2])

        new_action = copy.copy(cur_action)
        # print "Prev new action: ", new_action
        # print "Action Udate: ", last_action_update
        # print type(last_action_update)
        # print new_action[-1]
        new_action[-1] = last_action_update
        # print new_action
        return new_action


    '''
    Value gradients of action by finite difference.  
    '''
    def get_value_grad(self, cur_action): #best_seq: tuple (path sequence, path cost, reward, number of queries(called))
        # cur_action = cur_node.action
        # cur_reward = cur_node.reward
        # init_node = self.tree[cur_seq[0][:]]
        path_seq = []
        for seq in cur_action:
            path_seq.append(seq)
        # cur_node_reward = init_node[2]
        # num_queri = init_node[3]
        grad_x, grad_y = 0.0, 0.0
        # print(path_seq)
        if(len(path_seq)>=1):
            final_action = path_seq[-1]
            new_x_seq = copy.copy(path_seq[:])
            new_y_seq = copy.copy(path_seq[:])

            x = final_action[0]
            y = final_action[1]
            yaw = final_action[2]
            eps = 0.2
            step = 1.0
            x_dif = x + eps * step 
            y_dif = y + eps * step

            new_x_action = (x_dif, y, yaw)
            new_y_action = (x, y_dif, yaw)

            new_x_seq[-1] = new_x_action
            new_y_seq[-1] = new_y_action
            cur_reward = self.get_reward(path_seq)
            reward_x = self.get_reward(new_x_seq)
            # print "reward_x", reward_x
            reward_y = self.get_reward(new_y_seq)
            # print "Reward: ", reward_x, " ", reward_y
            # print "Cur reward: ", cur_reward 
            grad_x = (reward_x - cur_reward)/ eps
            grad_y = (reward_y - cur_reward) / eps
        return [grad_x, grad_y]

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

        if self.gradient_on and parent.depth <=2:
            print "##########################Grad Called################################"
            for i, action in enumerate(actions):
                current_action = actions[action]
                if len(current_action)>1:
                    # grad_val = self.get_value_grad(current_action)
                    grad_updated_action = self.update_action(current_action)
                    pg = Dubins_EqualPath_Generator(1,1.0, self.turning_radius, 2.0, self.ranges)
                    pg.cp = parent.pose
                    pg.goals = [grad_updated_action[-1]]
                    # print "Update Action: ", grad_updated_action
                    # print "Goals: ", pg.goals
                    new_action, new_dense_path = pg.make_sample_paths()
                    
                    new_action[0].append(grad_updated_action[-1])
                    new_dense_path[0].append(grad_updated_action[-1])
                    # print "New action: ", new_action
                    actions[action] = new_action[0]
                    dense_paths[action] = new_dense_path[0]

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



class cMCTS(MCTS):
    '''Class that establishes a MCTS for nonmyopic planning'''
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

