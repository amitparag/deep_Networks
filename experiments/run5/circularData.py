
"""
Generate data from the circumference of a circle

"""

import random
from collections import defaultdict
import numpy as np
import crocoddyl
crocoddyl.switchToNumpyArray()


def point(h:float= 0., k:float = 0., r:int = 2):
    """
    Generate random points from the circumference of a circle
    (h, k) -> center
    r  = radius
    """
    theta = random.random() * 2 * np.pi
    return h + np.cos(theta) * r, k + np.sin(theta) * r


def testData(ntraj:int = 50):
    """
    Data is in the form of (x, y, theta, cost)
    """
    
    model = crocoddyl.ActionModelUnicycle()
    cost_trajectory = []

    for _ in range(ntraj):
        x, y = point(0, 0, 2.1)    
        initial_config = [x, y, 0]               
        model.costWeights = np.matrix([1, 0.3]).T
        problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 30, model)
        ddp = crocoddyl.SolverFDDP(problem)
        ddp.solve([], [], 1000)
        if ddp.iter < 1000:
            a_ = np.array(ddp.xs)
            a = np.array(a_)
            b = a.flatten()
            c = np.append(a, sum(d.cost for d in ddp.datas()))
            cost_trajectory.append(c)
            
    return np.array(cost_trajectory)
