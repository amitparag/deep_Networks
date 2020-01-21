from keras import backend as K 
K.clear_session()
from keras import regularizers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

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
                     name:str = "baseline"
                    ):
    """
    2 hidden layers, sigmoid tanh
    
    """
    model = Sequential()
    model.add(Dense(NUNITS_INPUT, input_dim=(x_train.shape[1]), activation = "relu", name="input"))
    for _ in range(NHIDDEN):
        model.add(Dense(NUNITS,
                        activation = "tanh",
                        kernel_initializer='random_normal',
                        bias_initializer='random_uniform'                       
                        )) 

    model.add(Dense(y_train.shape[1],                    
                   activation = 'linear',
                   name = "output"))        

    sgd = optimizers.SGD(lr=lr, clipnorm=1.)
    adam = optimizers.Adam(lr = lr)
    model.compile(loss='mean_squared_error',
                  optimizer=adam,
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
