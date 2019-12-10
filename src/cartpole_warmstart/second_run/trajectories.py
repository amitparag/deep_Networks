
import os
from multiprocessing import Process, current_process, Pool
import numpy as np
import crocoddyl
crocoddyl.switchToNumpyArray()
import random
from cartpole import *
try:
   from six.moves import cPickle
except:
   import pickle as cPickle
import time

def get_trajectories(ntraj: int = 2):
    """
    In the cartpole system, each state is 4 dimensional, while the control is 1 dimensional.
    Therefore, the each row y_data should be 100 * 5 -> 500. So shape y_data : n_trjac X 500
    
    """
    initial_state = []
    trajectory = []
    for _ in range(ntraj):
        
        model = cartpole_model()

        x0 = [ random.uniform(0,1), random.uniform(-3.14, 3.14), random.uniform(0., 0.99)
                        , random.uniform(0., .99)]

        T  = 100
        problem = crocoddyl.ShootingProblem(np.matrix(x0).T, [ model ]*T, model)
        ddp = crocoddyl.SolverDDP(problem)
        # Solving this problem
        done = ddp.solve([], [], 100)
        del ddp.xs[0]

        y = [np.append(np.array(i).T, j) for i, j in zip(ddp.xs, ddp.us)]
        initial_state.append(x0)
        trajectory.append(np.array(y).flatten())
        #print(_)
        
    initial_state = np.asarray(initial_state)
    trajectory = np.asarray(trajectory)
    
    f = open('x_data.pkl', 'wb')
    cPickle.dump(initial_state, f, protocol=cPickle.HIGHEST_PROTOCOL)
    g = open("y_data.pkl", "wb")
    cPickle.dump(trajectory, g, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close(), g.close()
    print("Generated data")
    print(f"shape of x_train {initial_state.shape}, y_data {trajectory.shape}")
    
    
if __name__=='__main__':

    starttime = time.time()
    # Use all available cores
    pool = Pool(processes = 5)
    pool.map(get_trajectories, (10000,))
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))
 
