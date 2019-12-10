import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf

import numpy as np
import keras
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import crocoddyl
crocoddyl.switchToNumpyArray()
import random
from cartpole import *
from trajectories import *
try:
   from six.moves import cPickle
except:
   import pickle as cPickle
import time



def train_net(n_hidden: int = 5, traj:int = 1000,
	      save_model: bool = True, 
          data_present:bool= True):
    """
    A generic keras 4 hidden layered neural net with RMSprop as optimizer
    Each Layer has 256 neurons.
    
    """
    if data_present:
        with open('x_data.pkl', 'rb') as f: x_data = cPickle.load(f)
        with open('y_data.pkl', 'rb') as g: y_data = cPickle.load(g)
   
    else: x_data, y_data = get_trajectories(ntraj = traj)


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
        model.add(Dropout(0.05))
    model.add(Dense(y_data.shape[1], 
                    activation = 'linear'))        


    rms = optimizers.RMSprop(lr=0.001, rho=0.9)
    model.compile(loss='mean_squared_error',
                  optimizer=rms,
                  metrics=['mean_squared_error'])
    model.fit(x_data, 
                  y_data,
                  epochs = 200,
                  batch_size= 16,
                  verbose = 1
                  )
        
    
    if save_model:
        model.save_weights('model.h5')
        model_json = model.to_json()
        with open('model.json', "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        print("Saved model to disk")

    
    else: return model

#if __name__=="main":
starttime = time.time()
train_net()
print('Trained net in {} seconds'.format(time.time() - starttime))
