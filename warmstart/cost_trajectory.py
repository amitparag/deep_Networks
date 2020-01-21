
import random
import math
import numpy as np
import pandas as pd
import crocoddyl
crocoddyl.switchToNumpyArray()

def data(nTraj:int = 1000,
         fddp:bool = False
        ):
    model = crocoddyl.ActionModelUnicycle()
    x_data = []
    y_data = []
    
    for _ in range(nTraj):

        if thetaVal:
            initial_config = [random.uniform(-1.99, 1.99), random.uniform(-1.99, 1.99), random.uniform(0.,1.)]
        else:
            initial_config = [random.uniform(-1.99, 1.99), random.uniform(-1.99, 1.99), 0.0]
        model.costWeights = np.matrix([1, 0.3]).T
        problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 30, model)
        if fddp:
            ddp = crocoddyl.SolverFDDP(problem)
        else:
            ddp = crocoddyl.SolverDDP(problem)
            
        ddp.solve([], [], 1000)
        if ddp.iter < 1000:
            xs = np.array(ddp.xs)
            cost = []
            for d in ddp.data:
                cost.append(d.cost)
            cost = np.array(cost).reshape(31,1)
            state = np.hstack((cs, cost))
            
        y_data.append(state.ravel())   
        x_data.append(initial_config)
        
    x_data = np.array(x_data)
    y_data = np.array(y_data)
            
            
