"""
This script generates a number of optimal trajectories for unicycle in crocoddyl.
"""

import random
try:
   from six.moves import cPickle
except:
   import pickle as cPickle
import crocoddyl
import numpy as np
crocoddyl.switchToNumpyArray()
random.seed(20119)


class _Generator:
    """
    Abstract Class to generate trajectories from Crocoddyl for Unicycle

    """


    def generate_trajectories(self, trajectories: int, save: bool):
        raise NotImplementedError


class data(_Generator):
    """
    Child class of the abstract generator class
    """

    def __init__(self, n_trajectories: int = 10, state_weight: float = 1., control_weight: float = 0.3, nodes: int = 20):
        """
        @ Description:
           n_trajectories = number of trajectories
           state, control weights = Cost weight
           nodes = number of knot points
        """
        self.n_trajectories = n_trajectories
        self.state_weight = state_weight
        self.control_weight = control_weight
        self.knots = nodes


    def generate_trajectories(self, save: bool = False):
        """
        This could be done better with pool. But since we are generating a maximum of 10K trajectories, there' no need for pool
        """

        starting_configurations = []
        optimal_trajectories = []
        feasible_trajectories = 0
        for _ in range(self.n_trajectories):
            initial_config = np.matrix([random.uniform(-2.1, 2.), random.uniform(-2.1, 2.), random.uniform(0, 1)]).T
            model = crocoddyl.ActionModelUnicycle()
            model.costWeights = np.matrix([self.state_weight, self.control_weight]).T
            problem = crocoddyl.ShootingProblem(initial_config, [ model ] * self.knots, model)
            ddp = crocoddyl.SolverDDP(problem)
            ddp.solve()
            if ddp.isFeasible:
                ddp_xs = np.squeeze(np.asarray(ddp.xs))
                ddp_us = np.squeeze(np.asarray(ddp.us))
                feasible_trajectories += 1
                x = ddp_xs[1:,0]
                y = ddp_xs[1:,1]
                theta = ddp_xs[1:,2]
                velocity = ddp_us[:,0]
                torque = ddp_us[:,1]
                optimal_trajectory = np.hstack((x, y, theta, velocity, torque))
                starting_configurations.append(np.squeeze(np.asarray(initial_config)))
                optimal_trajectories.append(optimal_trajectory)

        starting_configurations = np.asarray(starting_configurations)
        optimal_trajectories = np.asarray(optimal_trajectories)
        if save:
            f = open('x_data.pkl', 'wb')
            cPickle.dump(starting_configurations, f, protocol=cPickle.HIGHEST_PROTOCOL)
            g = open("y_data.pkl", "wb")
            cPickle.dump(optimal_trajectories, g, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close(), g.close()


        else: 
            return starting_configurations, optimal_trajectories, feasible_trajectories



import os
from multiprocessing import Process, current_process

if __name__ =='__main__':
    data = data(10000)
    processes = []
    # Spawn 50 processes
    for _ in range(50):
        process = Process(target= data.generate_trajectories, args = (True,))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()


#starting_configurations, optimal_trajectories, correct = data.generate_trajectories(save = True)
