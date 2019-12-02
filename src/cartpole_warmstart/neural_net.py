# https://wltrimbl.github.io/2014-06-10-spelman/intermediate/python/04-multiprocessing.html
import os
from multiprocessing import Process, current_process
import numpy as np
import keras
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import crocoddyl
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
crocoddyl.switchToNumpyArray()
import random
from model_cartpole import *
try:
   from six.moves import cPickle
except:
   import pickle as cPickle
import timeit


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
        print(_)
        
    initial_state = np.asarray(initial_state)
    trajectory = np.asarray(trajectory)
    return initial_state, trajectory

def train_net(n_hidden: int = 4, traj:int = 1000, save_model: bool = False, save_data:bool = True):
    """
    A generic keras 4 hidden layered neural net with RMSprop as optimizer
    Each Layer has 256 neurons.
    
    """
    
    x_data, y_data = get_trajectories(ntraj = traj)
    if save_data:
        f = open('x_data.pkl', 'wb')
        cPickle.dump(x_data, f, protocol=cPickle.HIGHEST_PROTOCOL)
        g = open("y_data.pkl", "wb")
        cPickle.dump(y_data, g, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close(), g.close()
    print("Generated data") 
    print(f"shape of x_train {x_data.shape}, y_data {y_data.shape}")
    print("training")

    model = keras.models.Sequential()
    
    model.add(Dense(256, input_dim=(x_data.shape[1])))
    model.add(Activation('relu'))
    for _ in range(n_hidden):
                  
        model.add(Dense(256,
                        activation = "tanh",
                        kernel_initializer='random_uniform',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)))            
        model.add(Dropout(0.25))
    model.add(Dense(y_data.shape[1], 
                    activation = 'linear'))        


    rms = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    model.compile(loss='mean_squared_error',
                  optimizer=rms,
                  metrics=['mean_squared_error', "mean_absolute_error"])
    model.fit(x_data, 
                  y_data,
                  epochs = 200,
                  batch_size= 16,
                  verbose = 0
                  )
        
    #x_test, y_test = generate_data(1000) 
    #score = model.evaluate(x_test, y_test, batch_size = 16, use_multiprocessing=True)
    
    if save_model:
        model.save_weights('model.h5')
        model_json = model.to_json()
        with open('model.json', "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        print("Saved model to disk")

    
    else: return model
    
if __name__=='__main__':
    #import os
    #from multiprocessing import Process, current_process
    import numpy as np
    import timeit
    start = timeit.timeit()

    
    #processes = []
    #numbers = [1, 2, 3, 4]
    #numbers = range(1000)
    # Spawn 50 processes
    #for _ in range(50):
     #   process = Process(target=train_net, args = (5, 10000, True,True))
        #square(number)
      #  processes.append(process)
        # Processes are spawned by creating a process object and then calling its start method
       # process.start()


    # make use of .join method to make sure that all processes have finished before we run any further code
    #for process in processes:
     #   process.join()
    train_net(5, 1000, save_model=True, save_data = True)

    end = timeit.timeit()

    print(f"Done in {end - start}")
