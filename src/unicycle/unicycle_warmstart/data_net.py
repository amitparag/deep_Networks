
import random
import numpy as np
import crocoddyl
import keras
from keras.models import load_model
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
crocoddyl.switchToNumpyMatrix();

def generate_data(ntrajectories: int = 100000,
                  maxiter: int = 100,
                  knots: int = 20,
                  state_weight: float = 1., # can use random.uniform(1, 3)
                  control_weight: float = 0.3, # can use random.uniform(0,1)
                 ):
    """
    @ Arguments:
        ntrajectories : number of trajectories to generate
        maxiter       : number of iterations crocoddyl is supposed to make for each trajectory
        knots         : Knots, e.g T = 20, 30
        
        
    @ Returns:
        initial_states         : matrix of starting points, shape (ntrajectories, 3)....x_data
        optimal trajectories   : matrix of trajectories, shape (ntrajectores, knots).....y_data
                                 Each row of this matrix is:
                                     x1, y1, theta1, v1, w1, x2, y2, theta2, v2, w2, .....so on
        
    """
    starting_configurations = []
    optimal_trajectories = []
    
    for _ in range(ntrajectories):
            initial_config = [random.uniform(-2.1, 2.),
                                        random.uniform(-2.1, 2.), 
                                        random.uniform(0, 1)]
            
            model = crocoddyl.ActionModelUnicycle()
            model.costWeights = np.matrix([state_weight, control_weight]).T
            problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * knots, model)
            ddp = crocoddyl.SolverDDP(problem)
            ddp.solve([], [], maxiter)
            
            if ddp.isFeasible:
                state = np.array(ddp.xs)
                control = np.array(ddp.us)

                optimal_trajectory = np.hstack((state[1:,:], control))
                optimal_trajectory = np.ravel(optimal_trajectory)
                optimal_trajectories.append(optimal_trajectory)
                starting_configurations.append(initial_config)
                
    optimal_trajectories = np.array(optimal_trajectories)
    starting_configurations = np.array(starting_configurations) 
    return starting_configurations, optimal_trajectories
    


def train_net(n_hidden: int = 4, save_model: bool = False):
    """
    A generic keras 4 hidden layered neural net with RMSprop as optimizer
    Each Layer has 256 neurons.
    
    """
    x_data, y_data = generate_data()
    
    
    model = keras.models.Sequential()
    
    model.add(Dense(256, input_dim=(x_data.shape[1])))
    model.add(Activation('relu'))
    for _ in range(n_hidden):
        model.add(Dense(256, activation = "tanh")
                  
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
        model.save('basic_model.h5')
        
    else: return model    
if __name__=='__main__':
                  train_net(save_model=True)
