
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


def stateData(nTraj:int = 10000,
                    modelWeight1:float = 1,
                    modelWeight2:float = 0.4,
                    timeHorizon = 30,
                    initialTheta:float = 0.,
                    fddp:bool = False,
                    varyingInitialTheta:bool = False,
                    saveData:bool = False
                    ):

    trajectory = []
    model = crocoddyl.ActionModelUnicycle()

    for _ in range(nTraj):
        if varyingInitialTheta:
            initial_config = [random.uniform(-1.99, 1.99), random.uniform(-1.99, 1.99), random.uniform(0,1)]
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


            # Store trajectory(ie x, y, theta) 
            a = np.array(ddp.xs)

            # trajectory -> (x, y, theta)..(x, y, theta)..
            a = np.array(a.flatten())

            # append cost at the end of each trajectory
            b = np.append(a, sum(d.cost for d in ddp.datas()))
            trajectory.append(b)
        
    trajectory_ = np.array(trajectory)
    

    
    if saveData:
        df = pd.DataFrame(trajectory_[0:,0:])
        df.to_csv("trajectories.csv")
        


    print(trajectory_.shape)
    print("# Dataset 2: Shape (nTraj, 3 X TimeWindow + 1)..columns are recurring ie (x,y, theta),(x,y, theta).... \n\
           # The first two columns will form train. The last column is the cost associated with each trajectory")
                       
    return trajectory_

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
from utils_control import kerasBaselineNet

def neural_net(df, 
               NUNITS_INPUT = 64,
               NUNITS = 64,
               NHIDDEN = 3,
               lr = 1e-3,
               EPOCHS = 100,
               BATCHSIZE = 64,
               VERBOSE = 2,
               saveModel = False,
               scaler = 0,
               name = "trajectoryNet"):
    
    trajectory_dataset = df
    
    # Preprocessing trajectory_dataset
    if scaler == 0:
        
        trajectory_scaler = StandardScaler()
        trajectory_dataset = trajectory_scaler.fit_transform(trajectory_dataset)
        print(trajectory_dataset.shape)
        
    elif scaler == 1:
        trajectory_scaler = MinMaxScaler()
        trajectory_dataset = trajectory_scaler.fit_transform(trajectory_dataset)
        print(trajectory_dataset.shape)
        
    elif scaler == 2:
        trajectory_scaler = MaxAbsScaler()
        trajectory_dataset = trajectory_scaler.fit_transform(trajectory_dataset)
        print(trajectory_dataset.shape)
        
    elif scaler == -1:
        print("No scaling applied")
        print(trajectory_dataset.shape)
        

    # WITH trajectory DATASET
    x_train , y_train = trajectory_dataset[0:10000,0:3], trajectory_dataset[0:10000, 3:]
    print("TRAIN ",x_train.shape,",", y_train.shape)

    x_test , y_test = trajectory_dataset[10000:,0:3], trajectory_dataset[10000:, 3:]
    print("TEST ", x_test.shape,",", y_test.shape)

    trajectoryNet = kerasBaselineNet(x_train,
                y_train,
                x_test,
                y_test,
                NUNITS_INPUT = NUNITS_INPUT,
                NUNITS = NUNITS,
                NHIDDEN = NHIDDEN,
                lr = lr,
                EPOCHS = EPOCHS,
                BATCHSIZE = BATCHSIZE,
                VERBOSE = VERBOSE,
                saveModel = False,
                name = name)
    y_predicted = trajectoryNet.predict_on_batch(x_test)
    result_ = np.hstack((x_test, y_predicted))
    crocoddyl_ = np.hstack((x_train, y_train))
    
    if scaler != -1 :
        result = trajectory_scaler.inverse_transform(result_)
        crocoddyl = trajectory_scaler.inverse_transform(crocoddyl_)
    else: 
        result = result_
        crocoddyl = crocoddyl_

    return trajectoryNet


def testTrajectories(nTest:int = 50,
                    load_dataset:bool = False, 
                    load_net:bool = False):
    
    """
    1: get data set
    2: train neural net
    3: plot initial_state vs value function
    4: plot trajectories test
    5: warmstart
    """
    
    # Load dataset..............................................................
    if load_dataset:
        df_ = pd.read_csv("trajectories.csv")
        df = df_.values
    else:
        df = stateData(10000, saveData=True)
    
    # get neural_net...............................................................
    if load_net:
        net = load_model('trajectoryNet.h5')
        
    else:
        net = neural_net(df,
                NUNITS_INPUT = 64,
                NUNITS = 64,
                NHIDDEN = 3,
                lr = 1e-3,
                EPOCHS = 200,
                BATCHSIZE = 64,
                VERBOSE = 0,
                saveModel = True,
                scaler = -1,
                name = "trajectoryNet")
        

    # data for plotting
    trajectoryTEST, crocoddyl_data = generatePLOTDATA()
        
    # PLOT COROCODDYL...............................................................................     
    plotTrajectories(crocoddyl_data,crocoddyl = True ) 
    
    # Plotting neural net trajectories
    x_test , y_test = trajectoryTEST[0:,0:3], trajectoryTEST[0:, 3:]
    print("TEST ", x_test.shape,",", y_test.shape)
    y_predicted = trajectoryNet.predict_on_batch(x_test)
    result = np.hstack((x_test, y_predicted))
    # Now the predictions are in the shape [50, 94]
    cost_trajectoryTEST = defaultdict()

    for i in result:
        i = i.tolist()
        key = i.pop(-1)
        items = np.array(i).reshape(31, 3)
        # Delete theta
        np.delete(items, 2, 1)
        # Attach to key
        cost_trajectoryTEST[key] = items
        
        
    #PLOTTING NN PREDICTIONS............................................................................................
    plotTrajectories(cost_trajectoryTEST,crocoddyl = False ) 



if __name__ == '__main__':
    df = stateData(nTraj=20000)
    net = neural_net(df)
    
"""    

    
    
    
        
    
testTrajectories(50)        

"""
