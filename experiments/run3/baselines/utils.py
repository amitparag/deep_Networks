
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

crocoddyl.switchToNumpyArray()

import matplotlib.pyplot as plt
def point(h:float= 0., k:float = 0., r:int = 2):
    """
    Generate random points from the circumference of a circle
    (h, k) -> center
    r  = radius
    """
    theta = random.random() * 2 * np.pi
    return h + np.cos(theta) * r, k + np.sin(theta) * r

def rtPairs(radius:float = 2., npoints:int = 10):
    "Yield equidistant points from a circle"
    for i in range(len(radius)):
       for j in range(npoints[i]):    
        yield radius[i], j*(2 * np.pi / npoints[i])
        

def twoSpirals(n_points:int = 1000, noise:float = 0):
    """
     Returns the two spirals datasets.
     To bring it to between -2, 2, divide by 5,
     Usage x, y = twoSpiral(1000, 0)
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), 
            np.hstack((np.zeros(n_points),np.ones(n_points))))



def plotSpirals(npoints:int = 10000):
    
    X, y = twoSpirals(1000)

    plt.title('training set')
    plt.plot(X[y==0,0]/5, X[y==0,1]/5, '.', label='class 1', c = 'grey', )
    plt.plot(X[y==1,0]/5, X[y==1,1]/5, '.', label='class 2', c = 'green')
    plt.legend()
    plt.show()
    
def getSpiralPoints(nPoints:int = 10000):
    """
    Spiral points with a maximum radius of 2, centered at origin
    """
    X, y = twoSpirals(1000)
    a = X[y==0,0]/5
    b = X[y==0,1]/5
    coordinates = np.c_[a, b]
    return coordinates




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
            initial_config = [random.uniform(-1.99, 1.99), random.uniform(-1.99, 1.99), initialTheta]
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
        df_trajectory.to_csv("trajectory.csv")
        


    print(f"\nReturning {list(df.columns) } for {nTraj} trajectories.")
    print("\nReturing trajectories.")
    return df, df_trajectory
    
    
def kerasBaselineNet(x_train,
                     y_train,
                     x_test,
                     y_test,
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
    model.add(Dense(256, input_dim=(x_train.shape[1]), activation = "relu", name="input"))
    for _ in range(NHIDDEN):
        model.add(Dense(64,
                        activation = "tanh"
                        
                        )) 

    model.add(Dense(y_train.shape[1],
                    
                    activation = 'linear',
                   name = "output"))        

    sgd = optimizers.SGD(lr=lr, clipnorm=1.)
    model.compile(loss='mean_squared_error',
                  optimizer=sgd,
                  metrics=['mean_squared_error', "mean_absolute_error"])
    print(x_train.shape, y_train.shape)
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

