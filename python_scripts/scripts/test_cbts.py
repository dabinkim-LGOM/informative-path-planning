import grid_map_ipp_module as grid 
import obstacles as obs 
import numpy as np
import itertools
# import cv2 
import vis_grid_map as vis 
from Planning_Result import *
import GridMap_library as gd_lib 
import argparse 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", action="store", help="Environment of Exploration. Free, Box, or Harsh env.", default="Free")
    parse = parser.parse_args()

    '''
    Map Initialization
    1) Map size and obstacles
    2) Grid map for lidar sensor
    '''
    ### Map size & Obstacles
    map_max = 100.0
    ranges = (0.0, map_max, 0.0, map_max)
    
    ENVIRONMENT = parse.env 

    if(ENVIRONMENT=="Free"):
        obstacle_world = obs.FreeWorld()
        np_centers = []
        num_obs = 0
    elif(ENVIRONMENT=="Box"):
        block_x = 10.0
        block_y = 10.0
        center1, center2, center3, center4, center5 = (10.0, 30.0), (50.0, 80.0), (70., 20.), (50., 50.), (60., 70.)
        center6, center7, center8, center9, center10 = (30., 70.), (80., 50.), (40., 20.), (10., 90.), (90., 30.)
        centers = [center1, center2, center3, center4, center5, center6, center7, center8, center9, center10]
        # centers = [center1, center2]
        obstacle_world = obs.BlockWorld(extent = ranges, num_blocks=10, dim_blocks=(block_x, block_y), centers = centers )

        np_center1 = np.array([center1[0]-block_x/2.0, center1[1]-block_y/2.0, center1[0]+block_x/2.0, center1[1]+block_y/2.0  ])
        np_center2 = np.array([center2[0]-block_x/2.0, center2[1]-block_y/2.0, center2[0]+block_x/2.0, center2[1]+block_y/2.0  ])
        np_center3 = np.array([center3[0]-block_x/2.0, center3[1]-block_y/2.0, center3[0]+block_x/2.0, center3[1]+block_y/2.0  ])
        np_center4 = np.array([center4[0]-block_x/2.0, center4[1]-block_y/2.0, center4[0]+block_x/2.0, center4[1]+block_y/2.0  ])
        np_center5 = np.array([center5[0]-block_x/2.0, center5[1]-block_y/2.0, center5[0]+block_x/2.0, center5[1]+block_y/2.0  ])
        np_center6 = np.array([center6[0]-block_x/2.0, center6[1]-block_y/2.0, center6[0]+block_x/2.0, center6[1]+block_y/2.0  ])
        np_center7 = np.array([center7[0]-block_x/2.0, center7[1]-block_y/2.0, center7[0]+block_x/2.0, center7[1]+block_y/2.0  ])
        np_center8 = np.array([center8[0]-block_x/2.0, center8[1]-block_y/2.0, center8[0]+block_x/2.0, center8[1]+block_y/2.0  ])
        np_center9 = np.array([center9[0]-block_x/2.0, center9[1]-block_y/2.0, center9[0]+block_x/2.0, center9[1]+block_y/2.0  ])
        np_center10 = np.array([center10[0]-block_x/2.0, center10[1]-block_y/2.0, center10[0]+block_x/2.0, center10[1]+block_y/2.0  ])
        np_centers = [np_center1, np_center2, np_center3, np_center4, np_center5, np_center6, np_center7, np_center8, np_center9, np_center10]
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
    # grid_map = grid.ObstacleGridConverter(map_max, map_max, 2, np_centers)
    grid_map = grid.ObstacleGridConverter(map_max, map_max, num_obs, np_centers)
    
    raytracer = grid.Raytracer(map_max, map_max, num_obs, np_centers)

    '''World generation '''    
    
    # Options include mean, info_gain, and hotspot_info, mes'''
    reward_function = 'mean'

    world = Environment(ranges = ranges, # x1min, x1max, x2min, x2max constraints
                        NUM_PTS = 20, 
                        variance = 100.0, 
                        lengthscale = 3.0, 
                        visualize = False,
                        seed = 1)

    evaluation = Evaluation(world = world, 
                            reward_function = reward_function)

    # Gather some prior observations to train the kernel (optional)

    x1observe = np.linspace(ranges[0], ranges[1], 5)
    x2observe = np.linspace(ranges[2], ranges[3], 5)
    x1observe, x2observe = np.meshgrid(x1observe, x2observe, sparse = False, indexing = 'xy')  
    data = np.vstack([x1observe.ravel(), x2observe.ravel()]).T
    observations = world.sample_value(data)

    input_limit = [0.0, 10.0, -30.0, 30.0] #Limit of actuation 
    sample_number = 20 #Number of sample actions 

    '''
    Agent Initialization
    1) Initial pose 
    2) Lidar sensor construction 
    '''
    start_loc = (0.5, 0.5, 0.0)
    time_step = 100

    cur_x = 1.0
    cur_y = 1.0
    cur_yaw = 0.0
    pose = grid.Pose(cur_x, cur_y, cur_yaw)

    range_max, range_min, hangle_max, hangle_min, angle_resol, resol = 9.5, 0.5, 180.0, -180.0, 5.0, 1.0
    lidar = grid.Lidar_sensor(range_max, range_min, hangle_max, hangle_min, angle_resol, map_max, map_max, resol, raytracer)
    
    '''
    Planning Setup 
    '''
    display = True
    gradient_on = True

    gradient_step_list = [0.0, 0.05, 0.1, 0.15, 0.20]

    planning_type = 'nonmyopic'

    # for gradient_step in gradient_step_list:
    gradient_step = 0.0   
    print('range_max ' + str(range_max)+ ' iteration '+ ' gradient_step ' + str(gradient_step))
    iteration = 1
    planning = Planning_Result(planning_type, world, 'Free', obstacle_world, evaluation, reward_function, ranges, start_loc, input_limit, sample_number, time_step, grid_map, lidar, display, gradient_on, gradient_step, iteration)
    
    