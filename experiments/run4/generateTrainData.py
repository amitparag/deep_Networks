
import random
import math
import numpy as np
import pandas as pd
import crocoddyl
crocoddyl.switchToNumpyArray()

def getTrainingData(ntraj:int = 10000):

    trajectory = []
    
    for _ in range(ntraj):
        initial_config = [random.uniform(-1.99, 1.99), random.uniform(-1.99, 1.99), 0]            
        model = crocoddyl.ActionModelUnicycle()

        model.costWeights = np.matrix([1, 0.3]).T

        problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 30, model)
        ddp = crocoddyl.SolverFDDP(problem)
        ddp.solve([], [], 1000)
        if ddp.iter < 1000:

            # Store trajectory(ie x, y) 
            a = np.delete(np.array(ddp.xs), 2, 1)
            a = np.array(a.flatten())
            b = np.append(a, sum(d.cost for d in ddp.datas()))
            trajectory.append(b)
            #trajectory.append(sum(d.cost for d in ddp.datas()))


    # Dataset 2: Shape (number of init_points, 63)..columns are recurring (x,y). 
    # The first two columns will form train. The last column is the cost associated with each trajectory
    trajectory = np.array(trajectory)
    print(trajectory.shape)
    df_trajectory = pd.DataFrame(trajectory[0:,0:])
    

    return df_trajectory
