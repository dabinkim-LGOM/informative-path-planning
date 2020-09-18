'''
Baseline algorithm of Continuous Belief Tree Search (CBTS)
(IROS18 Continuous State-Action-Observation POMDPs
for Trajectory Planning with Bayesian Optimisation)
Global inner optimization of Bayesian Optimization for continuous action selection(dynamic sampling)
Optimization is based on particle swarm optimization. 
'''
import numpy as np
from safeopt import swarm as sw
# import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx 

class ParticleSwarmOpt(object):
    def __init__(self, t, sim_world, belief, acquisition_function, pose, x_bound, y_bound):
        self.time = t
        self.sim_world = sim_world
        # self.ranges = ranges
        self.belief = belief 
        self.acquisition_function = acquisition_function 
        self.x_bound = x_bound 
        self.y_bound = y_bound 
        self.bound = [(pose[0]-x_bound, pose[0]+x_bound), (pose[1]-y_bound, pose[1]+y_bound)]

    def fitness_function(self, particles):
        beta = 1.0 
        mean, var = self.belief.predict_values(particles)
        mean = mean.squeeze()
        std_dev = np.sqrt(var.squeeze())

        # lower_bound = np.atleast_1d(mean - beta * std_dev)
        # upper_bound = np.atleast_1d(mean - beta * std_dev)
        value = self.acquisition_function(time=self.time, xvals = particles, robot_model = sim_world)
        
        return value, True  

    def optimization(self):
        swarm_size = 100
        max_iter = 100 
        velocity = np.full([swarm_size,2], 1.)
        particles = np.full([swarm_size,2], self.pose) + 2*(np.random.rand(swarm_size, 2)-[0.5,0.5]) * [self.x_bound, self.y_bound]

        swarmopt = sw.SwarmOptimization(swarm_size, velocity, self.fitness_function, self.bound)
        swarmopt.init_swarm(particles)
        swarmopt.run_swarm(max_iter) 
        best_sol = swarmopt.global_best
        best_val, _ = self.fitness_function(best_sol)

        return best_sol, best_val 

