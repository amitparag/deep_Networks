

"""
This will define a neural net

"""
from keras import backend as K 
K.clear_session()
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
from extra_keras_utils import is_gpu_available
from keras.utils import plot_model



def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def max_error(y_true, y_pred):
    """
    Best score = 0
    """
    return K.max(K.abs(y_true - y_pred))


def kerasNet(x_data,
             y_data,
             NUNITS_INPUT = 64,
             NUNITS = 32,
             NHIDDEN = 2,
             lr = 1e-3,
             EPOCHS = 100,
             BATCHSIZE = 32,
             validation_split = 0.3,
             VERBOSE = 2,
             optimizer:str = "sgd",
             loss = ['mean_squared_error'],
             use_gpu:bool = True,
             saveModel: bool = False,
             plot_results:bool = True,
             baseline:bool = False                     
            ):
    """
    2 hidden layers, sigmoid tanh
    
    """
    if use_gpu:
        try:
            if is_gpu_available():
                print("Using gpu!")
            
        except: print("No GPU")  
            
    model = Sequential()
    model.add(Dense(NUNITS_INPUT, input_dim=(x_data.shape[1]), activation = "relu", name="First"))
    for _ in range(NHIDDEN):
        if baseline:
            model.name = "Baseline"
            model.add(Dense(NUNITS,
                      activation = "tanh"                           
                    ))           
        else:
            model.name = "Network1"
            model.add(Dense(NUNITS,
                        activation = "tanh",
                        kernel_initializer = 'random_uniform',
                        bias_initializer = 'random_normal',
                        kernel_regularizer=regularizers.l2(0.01),
                        activity_regularizer=regularizers.l1(0.01)    
                        )) 
            

    model.add(Dense(y_data.shape[1],                    
                   activation = 'linear',
                   name = "Final"))

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics = [rmse, max_error, 'mae']
                  )
        

    
    print("X_Train : ", x_data.shape," and Y_Train: ", y_data.shape)
    history = model.fit(x_data, 
                        y_data,
                        validation_split = validation_split,
                        epochs = EPOCHS,
                        batch_size= BATCHSIZE,
                        verbose = VERBOSE
                        ).history
    
    print(model.summary())
    plot_history(history)


    
    if saveModel:
        model.save(name + '.h5')
        print("Saved Model")
    return model
