'''
Baseline algorithm of Continuous Belief Tree Search (CBTS)
(IROS18 Continuous State-Action-Observation POMDPs
for Trajectory Planning with Bayesian Optimisation)
Global inner optimization of Bayesian Optimization for continuous action selection(dynamic sampling)
'''

class Baseline(object):
    def __init__(self, size):
        self.size = size