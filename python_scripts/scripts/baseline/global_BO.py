'''
Baseline algorithm of Continuous Belief Tree Search (CBTS)
(IROS18 Continuous State-Action-Observation POMDPs
for Trajectory Planning with Bayesian Optimisation)
Global inner optimization of Bayesian Optimization for continuous action selection(dynamic sampling)
Optimization is based on particle swarm optimization. 
'''
from __future__ import print_function, absolute_import, division
import numpy as np
import random 

from builtins import range


# import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx 

class ParticleSwarmOpt(object):
    def __init__(self, ranges, t, sim_world, belief, acquisition_function, pose, x_bound, y_bound):
        self.time = t
        self.ranges = ranges 
        self.sim_world = sim_world
        # self.ranges = ranges
        self.belief = belief 
        self.pose = pose 
        self.acquisition_function = acquisition_function 
        self.x_bound = x_bound 
        self.y_bound = y_bound 
        self.bound = [ ( max( pose[0]-x_bound, ranges[0]), min(pose[0] + x_bound, ranges[1]) ), (max(pose[1]-y_bound,ranges[2]), min(pose[1]+y_bound,ranges[3]))]


    def fitness_function(self, particles):
        beta = 1.0 
        # print(particles.shape)
        # mean, var = self.belief.predict_value(particles)
        # mean = mean.squeeze()
        # std_dev = np.sqrt(var.squeeze())

        # lower_bound = np.atleast_1d(mean - beta * std_dev)
        # upper_bound = np.atleast_1d(mean - beta * std_dev)
        values = np.empty(shape=(len(particles),1))
        safe = np.empty(shape=(len(particles),1), dtype=bool)
        
        for i, particle_pos in enumerate(particles):
            values[i] = self.acquisition_function(time=self.time, xvals = particle_pos, robot_model = self.belief)
            safe[i] = not (self.sim_world.in_obstacle(particle_pos) or (particle_pos[0] < self.ranges[0]) or particle_pos[0] > self.ranges[2] or particle_pos[1] < self.ranges[1] or particle_pos[1] > self.ranges[3])
        
        return values, safe 

    def optimization(self):
        swarm_size = 20
        max_iter = 20 
        velocity = np.full([1,2], 1.)
        # print()
        particles = [(random.uniform(self.bound[0][0], self.bound[0][1]), random.uniform(self.bound[1][0], self.bound[1][1])) for _ in range(swarm_size)]
        # particles = np.full([swarm_size,2], self.pose[0:1]) + 2*(np.random.rand(swarm_size, 2)-[0.5,0.5]) * [self.x_bound, self.y_bound]

        swarmopt = SwarmOptimization(swarm_size, velocity, self.fitness_function, self.bound)
        swarmopt.init_swarm(particles)
        swarmopt.run_swarm(max_iter) 
        best_sol = swarmopt.global_best
        # print("best_sol", best_sol)
        best_val, _ = self.fitness_function(np.reshape(best_sol, (1,2)))

        return best_sol, best_val 



class SwarmOptimization(object):
    """Constrained swarm optimization.

    Parameters 
    ----------
    swarm_size: int
        The number of particles
    velocity: ndarray
        The base velocities of particles for each dimension.
    fitness: callable
        A function that takes particles positions and returns two values. The
        first one corresponds to the fitness of the particle, while the second
        one is an array of booleans indicating whether the particle fulfills
        the constraints.
    bounds: list, optional
        A list of constraints to which particle exploration is limited. Of the
        form [(x1_min, x1_max), (x2_min, x2_max)...].
    """

    def __init__(self, swarm_size, velocity, fitness, bounds=None):
        """Initialization, see `SwarmOptimization`."""
        super(SwarmOptimization, self).__init__()

        self.c1 = self.c2 = 1
        self.fitness = fitness

        self.bounds = bounds
        if self.bounds is not None:
            self.bounds = np.asarray(self.bounds)

        self.initial_inertia = 1.0
        self.final_inertia = 0.1
        self.velocity_scale = velocity

        self.ndim = 2
        self.swarm_size = swarm_size

        self.positions = np.empty((swarm_size, 2), dtype=np.float)
        self.velocities = np.empty_like(self.positions)

        self.best_positions = np.empty_like(self.positions)
        self.best_values = np.empty(len(self.best_positions), dtype=np.float)
        self.global_best = None

    @property
    def max_velocity(self):
        """Return the maximum allowed velocity of particles."""
        return 10 * self.velocity_scale

    def init_swarm(self, positions):
        """Initialize the swarm.

        Parameters
        ----------
        positions: ndarray
            The initial positions of the particles.
        """
        self.positions = positions
        self.velocities = (np.random.rand(*self.velocities.shape) *
                           self.velocity_scale)

        values, safe = self.fitness(self.positions)

        # Initialize best estimates
        self.best_positions[:] = self.positions
        self.best_values = values

        self.global_best = self.best_positions[np.argmax(values), :]

    def run_swarm(self, max_iter):
        """Let the swarm explore the parameter space.

        Parameters
        ----------
        max_iter : int
            The number of iterations for which to run the swarm.
        """
        # run the core swarm optimization
        inertia = self.initial_inertia
        inertia_step = (self.final_inertia - self.initial_inertia) / max_iter

        for _ in range(max_iter):
            # update velocities
            delta_global_best = self.global_best - self.positions
            delta_self_best = self.best_positions - self.positions

            # Random update vectors
            r = np.random.rand(2 * self.swarm_size, self.ndim)
            r1 = r[:self.swarm_size]
            r2 = r[self.swarm_size:]

            # Update the velocities
            self.velocities *= inertia
            self.velocities += ((self.c1 * r1 * delta_self_best +
                                 self.c2 * r2 * delta_global_best) /
                                self.velocity_scale)

            inertia += inertia_step

            # clip
            # np.clip(velocities, -4, 4, out=velocities)
            np.clip(self.velocities,
                    -self.max_velocity,
                    self.max_velocity,
                    out=self.velocities)

            # update position
            self.positions += self.velocities

            # Clip particles to domain
            if self.bounds is not None:
                np.clip(self.positions,
                        self.bounds[:, 0],
                        self.bounds[:, 1],
                        out=self.positions)

            # compute fitness
            values, safe = self.fitness(self.positions)

            # find out which particles are improving
            update_set = values > self.best_values

            # print("values", values)
            # print("safety", safe)
            # update whenever safety and improvement are guaranteed
            update_set &= safe

            # print(values[update_set])
            # print(update_set)
            # print(self.best_positions)
            # print(self.positions)
            # print(self.positions[update_set])

            # print(self.best_positions[update_set])
            self.best_values[update_set] = values[update_set]
            # (self.best_positions[:,0])[update_set] = (self.positions[:,0])[update_set]
            # (self.best_positions[:,1])[update_set] = (self.positions[:,1])[update_set]
            for i in range(0,len(update_set)):
                if update_set[i]:
                    self.best_positions[i] = self.positions[i] 
            # self.best_positions[np.array(update_set)] = self.positions[np.array(update_set)]

            best_value_id = np.argmax(self.best_values)
            self.global_best = self.best_positions[best_value_id, :]

            if not any(update_set):
                break
