import numpy as np
import random
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import crocoddyl
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
crocoddyl.switchToNumpyMatrix()

class regression():
    def __init__(self,
                 n_trajectories: int = 10000,
                 n_hidden: int = 5,
                 state_weight: float = 1., 
                 control_weight: float = .3, 
                 nodes: int = 300,
                 save_trajectories: bool = False,
                 save_model: bool = False,
                 plot: bool = False):
        
        self.__n_trajectories = n_trajectories
        self.__state_weight = state_weight
        self.__control_weight = control_weight
        self.__nodes = nodes
        self.__n_hidden = n_hidden,
        self.save_trajectories = save_trajectories
        self.save_model = save_model
        self.plot = plot
        
        
    def _generate_trajectories(self):
        """
        This could be done better with pool. But since we are generating a maximum of 10K trajectories,
        there' no need for pool
        """

        starting_configurations = []
        optimal_trajectories = []
        feasible_trajectories = 0
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
                ddp_xs = np.squeeze(np.asarray(ddp.xs))
                ddp_us = np.squeeze(np.asarray(ddp.us))
                feasible_trajectories += 1
                x = ddp_xs[1:,0]
                y = ddp_xs[1:,1]
                theta = ddp_xs[1:,2]
                velocity = ddp_us[:,0]
                torque = ddp_us[:,1]
                optimal_trajectory = np.hstack((x, y, theta, velocity, torque))
                starting_configurations.append(np.squeeze(np.asarray(initial_config)))
                optimal_trajectories.append(optimal_trajectory)

        starting_configurations = np.asarray(starting_configurations)
        optimal_trajectories = np.asarray(optimal_trajectories)
        if self.save_trajectories:
            f = open('../../data/x_data.pkl', 'wb')
            cPickle.dump(starting_configurations, f, protocol=cPickle.HIGHEST_PROTOCOL)
            g = open("../../data/y_data.pkl", "wb")
            cPickle.dump(optimal_trajectories, g, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close(), g.close()



        return starting_configurations, optimal_trajectories, feasible_trajectories
    
    def keras_net(self, save: bool = False):
        starting_configurations, optimal_trajectories, feasible_trajectories = self._generate_trajectories()
        x_train = starting_configurations[0 : 9000, :]
        y_train = optimal_trajectories[0 : 9000, :]
        x_test = starting_configurations[9000 :, :]
        y_test = optimal_trajectories[9000 :, :]
        model = Sequential()
        model.add(Dense(256, input_dim=(starting_configurations.shape[1])))
        model.add(Activation('relu'))
        for _ in range(5):
            model.add(Dense(256,
                            activation = "tanh",
                            kernel_initializer='random_uniform',
                            kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1(0.01)))            
            model.add(Dropout(0.25))
            
        model.add(Dense(optimal_trajectories.shape[1], 
                        activation = 'linear'))
        
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['mean_squared_error', "mean_absolute_error"])
        
        print('Train...')
        
        model.fit(x_train, 
                  y_train,
                  batch_size = 32,
                  epochs = 200,
                  verbose = 1
                  )
        
        score = model.evaluate(x_test, y_test, batch_size = 16, use_multiprocessing=True)
        
        print(score)
        
if __name__=='__main__':
        regression = regression()
        regression.keras_net()
