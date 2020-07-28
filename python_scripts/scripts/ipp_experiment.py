# !/usr/bin/python

import os
import time
import sys
import logging 
import numpy as np 
import argparse 

import grid_map_ipp_module as grid 
import obstacles as obs 
import vis_grid_map as vis 
from Planning_Result import *
import GridMap_library as gd_lib 


#Commaind Line Options
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", action="store", type=int, help="Random seed for environment generation.", default=0)
parser.add_argument("-r", "--reward", action="store", help="Reward function. Should be mes, ei, info_gain, mean", default="mean")
parser.add_argument("-p", "--pathset", action="store", help="Action set type. Dubins, or conti_sampler", default="dubins")
parser.add_argument("-n", "--nonmyopic", action="store", help="myopic, nonmyopic, or coverage", default="nonmyopic")
parser.add_argument("-g", "--goal", action="store_true", help="Consider the reward of final point only if flag set.", default=False)
parser.add_argument("-e", "--env", action="store", help="Environment of Exploration. Free, Box, or Harsh env.", default="Free")
parser.add_argument("-z", "--size", action="store", type=float, help="Size of map environment. 50, 100, 200", default=100.)
parser.add_argument("-d", "--gradient", action="store", type=float, help="Gradient step", default=0.0)

#parser commaind line options
parse = parser.parse_args()

#Read command line options
SEED = parse.seed
REWARD_FUNCTION = parse.reward
PATHSET = parse.pathset
PLANNER = parse.nonmyopic
GOAL_ONLY = parse.goal
ENVIRONMENT = parse.env 
SIZE = parse.size 
GRAD_STEP = parse.gradient 

MIN_COLOR = -25.
MAX_COLOR = 25. 

 # Set up paths for logging the data from the simulation run
if not os.path.exists('./figures/' + str(REWARD_FUNCTION)):
    os.makedirs('./figures/' + str(REWARD_FUNCTION))
logging.basicConfig(filename = './figures/'+ REWARD_FUNCTION + '/robot.log', level = logging.INFO)
logger = logging.getLogger('robot')


#Set Environment
map_max = SIZE 
ranges = (0.0, map_max, 0.0, map_max)

if(ENVIRONMENT=="Free"):
    obstacle_world = obs.FreeWorld()
    np_centers = []
    num_obs = 0
elif(ENVIRONMENT=="Box"):
    block_x = 10.0
    block_y = 10.0
    center1, center2, center3, center4, center5 = (10.0, 30.0), (50.0, 80.0), (70., 20.), (50., 50.), (60., 70.)
    centers = [center1, center2]
    obstacle_world = obs.BlockWorld(extent = ranges, num_blocks=5, dim_blocks=(block_x, block_y), centers = centers )

    np_center1 = np.array([center1[0]-block_x/2.0, center1[1]-block_y/2.0, center1[0]+block_x/2.0, center1[1]+block_y/2.0  ])
    np_center2 = np.array([center2[0]-block_x/2.0, center2[1]-block_y/2.0, center2[0]+block_x/2.0, center2[1]+block_y/2.0  ])
    np_center3 = np.array([center3[0]-block_x/2.0, center3[1]-block_y/2.0, center3[0]+block_x/2.0, center3[1]+block_y/2.0  ])
    np_center4 = np.array([center4[0]-block_x/2.0, center4[1]-block_y/2.0, center4[0]+block_x/2.0, center4[1]+block_y/2.0  ])
    np_center5 = np.array([center5[0]-block_x/2.0, center5[1]-block_y/2.0, center5[0]+block_x/2.0, center5[1]+block_y/2.0  ])
    np_centers = [np_center1, np_center2, np_center3, np_center4, np_center5]
    num_obs = 5
elif(ENVIRONMENT=="Harsh"):
    block_x = 10.0
    block_y = 47.0
    center1, center2 = (50.0, 77.0), (50.0, 22.0)
    centers = [center1, center2]
    obstacle_world = obs.BlockWorld(extent = ranges, num_blocks=5, dim_blocks=(block_x, block_y), centers = centers )
    
    np_center1 = np.array([center1[0]-block_x/2.0, center1[1]-block_y/2.0, center1[0]+block_x/2.0, center1[1]+block_y/2.0  ])
    np_center2 = np.array([center2[0]-block_x/2.0, center2[1]-block_y/2.0, center2[0]+block_x/2.0, center2[1]+block_y/2.0  ])
    np_centers = [np_center1, np_center2]
    num_obs = 2
else:
    raise NameError('Invalid Environment Type')

### Grid Map
grid_map = grid.ObstacleGridConverter(map_max, map_max, num_obs, np_centers)
raytracer = grid.Raytracer(map_max, map_max, num_obs, np_centers)


'''World generation '''    

# Options include mean, info_gain, and hotspot_info, mes'''
reward_function = REWARD_FUNCTION

world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                    NUM_PTS = 20, 
                    variance = 100.0, 
                    lengthscale = 3.0, 
                    visualize = False,
                    seed = SEED)

evaluation = Evaluation(world = world, 
                        reward_function = reward_function)

# Gather some prior observations to train the kernel (optional)

x1observe = np.linspace(ranges[0], ranges[1], 5)
x2observe = np.linspace(ranges[2], ranges[3], 5)
x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
observations = world.sample_value(data)

input_limit = [0.0, 10.0, -30.0, 30.0] #Limit of actuation 
sample_number = 10 #Number of sample actions 

'''
Agent Initialization
1) Initial pose 
2) Lidar sensor construction 
'''
start_loc = (0.5, 0.5, 0.0)
time_step = 200

cur_x = 1.0
cur_y = 1.0
cur_yaw = 0.0
pose = grid.Pose(cur_x, cur_y, cur_yaw)

range_max, range_min, hangle_max, hangle_min, angle_resol, resol = 4.5, 0.5, 180.0, -180.0, 5.0, 1.0
lidar = grid.Lidar_sensor(range_max, range_min, hangle_max, hangle_min, angle_resol, map_max, map_max, resol, raytracer)

'''
Planning Setup 
'''
display = False
gradient_on = True

# gradient_step_list = [0.0, 0.05, 0.1, 0.15, 0.20]

planning_type = PLANNER

# for gradient_step in gradient_step_list:
gradient_step = GRAD_STEP    
print('range_max ' + str(ranges[1])+ ' iteration '+ ' gradient_step ' + str(gradient_step))
iteration = 1

planning = Planning_Result(planning_type, world, ENVIRONMENT, obstacle_world, evaluation, reward_function, ranges, start_loc, input_limit, sample_number, time_step,
                        grid_map, lidar, display, gradient_on, gradient_step, iteration)

