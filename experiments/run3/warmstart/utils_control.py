
"""
Generate data.

"""
import random
import math
import numpy as np
import pandas as pd
import crocoddyl
from keras import backend as K 
K.clear_session()
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
crocoddyl.switchToNumpyArray()






def stateData(nTraj:int = 10,
                    modelWeight1:float = 1,
                    modelWeight2:float = 0.4,
                    timeHorizon = 30,
                    initialTheta:float = 0.,
                    fddp:bool = False,
                    varyingInitialTheta:bool = False,
                    saveData:bool = False
                    ):
    data = []
    trajectory = []
    model = crocoddyl.ActionModelUnicycle()

    for _ in range(nTraj):
        if varyingInitialTheta:
            initial_config = [random.uniform(-1.99, 1.99), random.uniform(-1.99, 1.99), random.uniform(0.,1.)]
        else:
            initial_config = [random.uniform(-1.99, 1.99), random.uniform(-1.99, 1.99), initialTheta] 
        model.costWeights = np.matrix([modelWeight1, modelWeight2]).T
        problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * timeHorizon, model)
        if fddp:
            ddp = crocoddyl.SolverFDDP(problem)
        else:
            ddp = crocoddyl.SolverDDP(problem)
        # Will need to include a assertion check for done in more complicated examples
        ddp.solve([], [], 1000)
        if ddp.iter < 1000:

            state = []
            # Attach x, y, theta
            state.extend(i for i in ddp.xs[0])
            # Attach control: linear_velocity, angular_velocty
            state.extend(i for i in ddp.us[0])
            # Attach value function: cost
            state.append(sum(d.cost for d in ddp.datas()))
            # Attach the number of iterations
            state.append(ddp.iter)
            data.append(state)
            
            
            # Store trajectory(ie x, y) 
            a = np.delete(np.array(ddp.xs), 2, 1)
            a = np.array(a.flatten())
            b = np.append(a, sum(d.cost for d in ddp.datas()))
            trajectory.append(b)
            

    # Dataset 1: x, y, z, linear_velocity, angular_velocity, value_function, iterations        
    data = np.array(data)
    df = pd.DataFrame(data[0:,0:], columns = ["x_position", "y_position",
                                              "z_position", "linear_velocity",
                                              "angular_velocity", "value_function",
                                              "iterations"])
    df.drop(['z_position'], axis=1, inplace=True)

    # Dataset 2: Shape (number of init_points, 63)..columns are recurring (x,y). 
    # The first two columns will form train.
    # The last column is the cost associated with each trajectory
    trajectory = np.array(trajectory)
    print(trajectory.shape)
    print("# Dataset 2: Shape (number of init_points, 63)..columns are recurring ie (x,y),(x,y).... \n\
           # The first two columns will form train. The last column is the cost associated with each trajectory")
    df_trajectory = pd.DataFrame(trajectory[0:,0:])
                       

    
    if saveData:
        df.to_csv("initialStates.csv")
        


    print(f"\nReturning {list(df.columns) } for {nTraj} trajectories.")
    print("\nReturing trajectories.")
    return df
    
def kerasBaselineNet(x_train,
                     y_train,
                     x_test,
                     y_test,
                     NUNITS_INPUT = 32,
                     NUNITS = 32,
                     NHIDDEN = 2,
                     lr = 1e-3,
                     EPOCHS = 100,
                     BATCHSIZE = 64,
                     VERBOSE = 2,
                     saveModel: bool = False,
                     name:str = "control"
                    ):
    """
    2 hidden layers, sigmoid tanh
    
    """
    model = Sequential()
    model.add(Dense(NUNITS_INPUT, input_dim=(x_train.shape[1]), activation = "relu", name="input"))
    for _ in range(NHIDDEN):
        model.add(Dense(NUNITS,
                        activation = "tanh"
                        
                        )) 

    model.add(Dense(y_train.shape[1],
                    
                    activation = 'linear',
                   name = "output"))        

    sgd = optimizers.SGD(lr=lr, clipnorm=1.)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mean_squared_error', "mean_absolute_error"])
    #print(x_train.shape, y_train.shape)
    model.fit(x_train, 
              y_train,
              epochs = EPOCHS,
              batch_size= BATCHSIZE,
              verbose = VERBOSE
             )
    print(model.summary())
    # evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
    
    if saveModel:
        model.save(name + '.h5')    
        print("Saved Model")
    return model

