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

class Baseline(object):
    def __init__(self, size, belief, aquisition_function):
        self.size = size

    def optimization(self):
        
        swarmopt = sw.SwarmOptimization(swarm_size, velocity, fitness,bound)
