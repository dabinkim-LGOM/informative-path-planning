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
# import gpmodel_library as gp_lib
# from continuous_traj import continuous_traj

from Environment import *
from Evaluation import *
from GPModel import *
import GridMap_library as sdflib


class MCTS():
    '''Class that establishes a MCTS for nonmyopic planning'''
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, planning_limit, frontier_size,
                path_generator, aquisition_function, time, gradient_on, grad_step, lidar, SFC):
        '''Initialize with constraints for the planning, including whether there is 
           a budget or planning horizon
           budget - length, time, etc to consider
           belief - GP model of the robot current belief state
           initial_pose - (x,y,rho) for vehicle'''
        self.ranges = ranges
        self.budget = computation_budget
        self.GP = belief
        self.cp = initial_pose
        self.limit = planning_limit
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

        while time.clock() - time_start < self.budget:
            current_node = self.tree_policy() #Find maximum UCT node (which is leaf node)
            
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
    
    def print_self(self):
        print(self.name)

class Tree(object):
    def __init__(self, f_rew, f_aqu,  belief, pose, path_generator, t, depth, param, c):
        self.path_generator = path_generator
        self.max_depth = depth #Maximum rollout depth?
        self.param = param
        self.t = t
        self.f_rew = f_rew
        self.aquisition_function = f_aqu
        self.c = c

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
        next_leaf, reward = self.leaf_helper(self.root, reward = 0.0,  belief = belief) 
        #print "Next leaf:", next_leaf
        #print "Reward:", reward
        self.backprop(next_leaf, reward)

    def leaf_helper(self, current_node, reward, belief):
        if current_node.node_type == 'B':
            # Root belief node
            if current_node.depth == self.max_depth:
                #print "Returning leaf node:", current_node.name, "with reward", reward
                return current_node, reward
            # Intermediate belief node
            else:
                if current_node.children is None:
                    self.build_action_children(current_node)

                # If no viable actions are avaliable
                if current_node.children is None:
                    return current_node, reward

                child = self.get_next_child(current_node)
                #print "Selecting next action child:", child.name

                # Recursive call
                return self.leaf_helper(child, reward, belief)

        # At random node, after selected action from a specific node
        elif current_node.node_type == 'BA':
            # Copy old belief
            #gp_new = copy.copy(current_node.belief) 
            #gp_new = current_node.belief

            # Sample a new set of observations and form a new belief
            #xobs = current_node.action
            obs = np.array(current_node.action)
            print(obs)
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
                nchild = len(current_node.children)
                #print "Current depth:", current_node.depth, "alpha:", alpha
                #print "First:", np.floor(nchild ** alpha)
                #print "Second:", np.floor((nchild - 1) ** alpha)

                '''
                Progressive Widening 
                '''
                if current_node.depth < self.max_depth - 1 and np.floor(nchild ** alpha) == np.floor((nchild - 1) ** alpha):
                    #print "Choosing from among current nodes"
                    #child = random.choice(current_node.children)
                    #print "number quieres:", nqueries
                    child = random.choice(current_node.children)
                    nqueries = [node.nqueries for node in current_node.children]
                    #select node which has the minimum queuried number & random if ties 
                    child = random.choice([node for node in current_node.children if node.nqueries == min(nqueries)])

                    if True:
                        belief.add_data(xobs, child.zvals)
                    #print "Selcted child:", child.nqueries
                    return self.leaf_helper(child, reward + r, belief)

            if True:
                if belief.model is None:
                    n_points, input_dim = xobs.shape
                    zmean, zvar = np.zeros((n_points, )), np.eye(n_points) * belief.variance
                    zobs = np.random.multivariate_normal(mean = zmean, cov = zvar)
                    zobs = np.reshape(zobs, (n_points, 1))
                else:
                    zobs = belief.posterior_samples(xobs, full_cov = False, size = 1)
                    n_points, input_dim = xobs.shape
                    zobs = np.reshape(zobs, (n_points,1))

                # print(xobs)
                # print(type(zobs))
                belief.add_data(xobs, zobs) # Work as continuous observation 

            else:
                zobs = belief.posterior_samples(xobs, full_cov = False, size = 1)
                n_points, input_dim = xobs.shape
                zobs = np.reshape(zobs, (n_points,1))

            belief.add_data(xobs, zobs) #If we add tree's belief the new query points, it means it is continuous-observation case. 
            pose_new = current_node.dense_path[-1]
            child = Node(pose = pose_new, 
                         parent = current_node, 
                         name = current_node.name + '_belief' + str(current_node.depth + 1), 
                         action = None, 
                         dense_path = None, 
                         zvals = zobs)
            #print "Adding next belief child:", child.name
            current_node.add_children(child)

            # Recursive call
            return self.leaf_helper(child, reward + r, belief)

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
        # actions = self.path_generator.get_path_set(parent.pose)
        # dense_paths = [0]
        if len(actions) == 0:
            print("No actions!")
            return
        
        #print "Creating children for:", parent.name
        for i, action in enumerate(actions.keys()):
            #print "Action:", i
            parent.add_children(Node(pose = parent.pose, 
                                    parent = parent, 
                                    name = parent.name + '_action' + str(i), 
                                    action = actions[action], 
                                    dense_path = dense_paths[action],
                                    zvals = None))

    def print_tree(self):
        counter = self.print_helper(self.root)
        print "# nodes in tree:", counter

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
    def __init__(self, ranges, obstacle_world, computation_budget, belief, initial_pose, planning_limit, frontier_size,
                path_generator, aquisition_function, time, gradient_on, grad_step, lidar, SFC):
        # Call the constructor of the super class
        super(cMCTS, self).__init__(ranges, obstacle_world, computation_budget, belief, initial_pose, planning_limit, frontier_size,
                                    path_generator, aquisition_function, time, gradient_on, grad_step, lidar, SFC)
        # self.tree_type = tree_type
        # Tree type is dpw 
        # self.aq_param = aq_param
        # self.GP = belief

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
        print "Setting c to :", self.c

    def choose_trajectory(self):
        #Main function loop which makes the tree and selects the best child
        #Output: path to take, cost of that path

        # randomly sample the world for entropy search function
        if self.f_rew == 'mes':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t = self.t, visualize=True)
            param = (self.max_val, self.max_locs, self.target)
            print("Hello")
        elif self.f_rew == 'exp_improve':
            param = [self.current_max]
        elif self.f_rew == 'naive' or self.f_rew == 'naive_value':
            self.max_val, self.max_locs, self.target  = sample_max_vals(self.GP, t= self.t, nK=int(self.aq_param[0]), visualize=True, f_rew=self.f_rew)
            param = ((self.max_val, self.max_locs, self.target), self.aq_param[1])
        else:
            param = None

        # initialize tree
        if self.tree_type == 'dpw':
            self.tree = Tree(self.f_rew, self.aquisition_function, self.GP, self.cp, self.path_generator, self.t, depth = self.rl, param = param, c = self.c)
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
        print "Rollouts completed in", str(time_end - time_start) +  "s"
        print "Number of rollouts:", i
        self.tree.print_tree()

        print [(node.nqueries, node.reward/(node.nqueries+0.1)) for node in self.tree.root.children]

        best_child = self.tree.root.children[np.argmax([node.nqueries for node in self.tree.root.children])]
        # best_child = random.choice([node for node in self.tree.root.children if node.nqueries == max([n.nqueries for n in self.tree.root.children])])
        all_vals = {}
        for i, child in enumerate(self.tree.root.children):
            all_vals[i] = child.reward / (float(child.nqueries)+0.1)
            # print(str(i) + " is " + str(all_vals[i]))

        paths, dense_paths = self.path_generator.get_path_set(self.cp)
        return best_child.action, best_child.dense_path, best_child.reward/(float(best_child.nqueries)+1.0), paths, all_vals, self.max_locs, self.max_val, self.target

        # get the best action to take with most promising futures, base best on whether to
        # consider cost
        #best_sequence, best_val, all_vals = self.get_best_child()

        #Document the information
        #print "Number of rollouts:", i, "\t Size of tree:", len(self.tree)
        #logger.info("Number of rollouts: {} \t Size of tree: {}".format(i, len(self.tree)))
        #np.save('./figures/' + self.f_rew + '/tree_' + str(t) + '.npy', self.tree)
        #return self.tree[best_sequence][0], self.tree[best_sequence][1], best_val, paths, all_vals, self.max_locs, self.max_val

