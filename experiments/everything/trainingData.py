

import random
import math
import numpy as np
import pandas as pd
import crocoddyl
crocoddyl.switchToNumpyArray()

def getTrainingData(nTraj:int = 10000, thetaVal:bool = False, fddp:bool = False):
    model = crocoddyl.ActionModelUnicycle()
    
    trajectory = []
    
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
            state = []

            # Attach x, y, theta. this will form the columns in x_training... index 0, 1, 2
            state.extend(i for i in ddp.xs[0])
            # Attach control: linear_velocity, angular_velocty.....index 3, 4
            state.extend(i for i in ddp.us[0])
            # Attach value function: cost....index 5
            state.append(sum(d.cost for d in ddp.datas()))
            
            
            trajectory_control = np.hstack((np.array(ddp.xs[1:]), np.array(ddp.us)))
            state.extend(i for i in trajectory_control.flatten()) #..............index 6 to -1.....30 X 5
        trajectory.append(state)
            
    data = np.array(trajectory)
    df = pd.DataFrame(data)
    #....................................
    #    x_train = data[:,0:3]  x,y,z
    #    y_train = data[:,3:]   lv, av, cost, (x, y, theta, control1, control2)
    #....................................    
    return data
                                              
        

            
