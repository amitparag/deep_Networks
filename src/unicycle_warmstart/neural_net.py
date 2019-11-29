
import numpy as np
import random
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import crocoddyl
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
crocoddyl.switchToNumpyArray()
random.seed(1)

        
class train_net():
    """
    A self contained class to generate trajectories to train the neural net and warmstart the solver.
    """
    def __init__(self,
                 n_trajectories: int = 10000, 
                 state_weight: float = 1., # can use random.uniform(1, 3)
                 control_weight: float = 0.3, # can use random.uniform(0,1)
                 nodes: int = 20,
                 n_hidden: int = 5,
                 neurons: int = 256,
                 optimizer: str = 'rms',
                 save_trajectories: bool = False,
                 save_model: bool = False,
                 plot: bool = False):
        
        """
        @ Args:
             n_trajectories : number of trajectories to generate from crocoddyll
             
             state_weight   : the weight of state in unicycle
             
             control_weight : control weight of the unicycle
             
             nodes          : number of knots, e.g T = 10 or 30   
             
             n_hidden       : number of hidden layers in the neural network
             
             neurons        : number of neurons in the hidden layer
             
             optmizer       : the optimizer to be used to train the net
             
             save_trajectories : save the trajectories with pickle
             
             save_model     : save the net
             
             plot           : plot results  
             
        """
        self.__n_trajectories = n_trajectories
        self.__state_weight = state_weight
        self.__control_weight = control_weight
        self.__nodes = nodes
        self.__n_hidden = n_hidden
        self.__neurons = neurons
        self.optimizer = optimizer
        self.save_trajectories = save_trajectories
        self.save_model = save_model
        self.plot = plot

    def generate_trained_net(self):
        """
        This could be done better with pool. But since we are generating a maximum of 10K trajectories, 
        there' no need for pool.
        
        @ Description: generate 10K trajectories, each trajectory with same state and control weight.
        """

        starting_configurations = []
        optimal_trajectories = []
        
        for _ in range(self.__n_trajectories):
            initial_config = np.matrix([random.uniform(-2.1, 2.),
                                        random.uniform(-2.1, 2.), 
                                        random.uniform(0, 1)]).T
            
            model = crocoddyl.ActionModelUnicycle()
            model.costWeights = np.matrix([self.__state_weight, self.__control_weight]).T
            problem = crocoddyl.ShootingProblem(initial_config, [ model ] * self.__nodes, model)
            ddp = crocoddyl.SolverDDP(problem)
            ddp.solve()
            if ddp.isFeasible:
                state = np.matrix(ddp.xs)
                control = np.matrix(ddp.us)

                optimal_trajectory = np.hstack((state[1:,:], control))
                optimal_trajectory = np.ravel(optimal_trajectory)
                optimal_trajectories.append(optimal_trajectory)  

                """
                So the data set will have self.__nodes * 5 as number of features
                
                
                """

                starting_configurations.append(ddp.xs[0])
                
                #print(optimal_trajectories)

        optimal_trajectories = np.array(optimal_trajectories)
        starting_configurations = np.array(starting_configurations)
        print(optimal_trajectories.shape)
        print(starting_configurations.shape)
        
        if self.save_trajectories:
            f = open('x_data.pkl', 'wb')
            cPickle.dump(starting_configurations, f, protocol=cPickle.HIGHEST_PROTOCOL)
            g = open("y_data.pkl", "wb")
            cPickle.dump(optimal_trajectories, g, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close(), g.close()
            
        

        x_train = starting_configurations[0 : 9000, :]
        y_train = optimal_trajectories[0 : 9000, :]
        x_test = starting_configurations[9000 :, :]
        y_test = optimal_trajectories[9000 :, :]
        print(f'Train size {x_train.shape}, {y_train.shape}, test size {x_test.shape}, {y_test.shape}')
        model = Sequential()
        model.add(Dense(256, input_dim=(starting_configurations.shape[1])))
        model.add(Activation('relu'))
        for _ in range(self.__n_hidden):
            model.add(Dense(256,
                            activation = "tanh",
                            kernel_initializer='random_uniform',
                            kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1(0.01)))            
            model.add(Dropout(0.25))
            
        model.add(Dense(optimal_trajectories.shape[1], 
                        activation = 'linear'))        
        
     
        rms = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
   
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
        if self.optimizer == 'sgd':
            model.compile(loss='mean_squared_error',
                      optimizer=sgd,
                      metrics=['mean_squared_error', "mean_absolute_error"])
            
        else :
            model.compile(loss='mean_squared_error',
                      optimizer=rms,
                      metrics=['mean_squared_error', "mean_absolute_error"])    
        
        print(f'Training neural net on {self.__n_trajectories}...')
        
        model.fit(x_train, 
                  y_train,
                  epochs = 200,
                  batch_size= 16,
                  verbose = 0
                  )
        
        score = model.evaluate(x_test, y_test, batch_size = 16, use_multiprocessing=True)
        
        #print(score)
        print(self.__nodes)
        if self.save_model:
            model.save('model.h5')  
            
        
        return model

        
        
if __name__=='__main__':
    net = train_net(save_model = False)
    neuralNet = net.generate_trained_net()
