
from keras.models import model_from_json
from net import *
from keras import optimizers
import numpy as np
import crocoddyl
crocoddyl.switchToNumpyArray()
import random
from cartpole import *

rms = optimizers.RMSprop(lr=0.001, rho=0.9)
json_file = open('model_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
neuralNet = model_from_json(loaded_model_json)
neuralNet.load_weights("model_new.h5")
print("Loaded model from disk")
neuralNet.compile(loss='mean_squared_error',
                  optimizer=rms,
                  metrics=['mean_squared_error'])



for _ in range(0, 1000):

    x0 = [ random.uniform(0,1), random.uniform(-3.14, 3.14), random.uniform(0., 0.99)
                    , random.uniform(0., .99)]


    model = cartpole_model()
    prediction = neuralNet.predict(np.array(x0).reshape(1,4))
    prediction = prediction.reshape(100, 5)
    ##Every fifth element is control, the first 4 elements are state
    ddp_xs, ddp_us = [], []
    ddp_xs.append(np.matrix(x0).T)
    xs = prediction[:,0:4]
    us = prediction[:,4]

    for _ in range(xs.shape[0]):
        ddp_xs.append(np.matrix(xs[_]).T)

    for _ in  range(us.shape[0]):
        ddp_us.append(np.matrix(us[_]).T)

    T  = 100
    problem = crocoddyl.ShootingProblem(np.matrix(x0).T, [ model ]*T, model)
    ddp = crocoddyl.SolverDDP(problem)
    # Solving this problem
    done = ddp.solve(ddp_xs, ddp_us)
    print("Warmstarting " , ddp.iter)

    del model, prediction, ddp_xs, ddp_us, ddp, problem, T

    model2 = cartpole_model()
    T  = 100

    problem2 = crocoddyl.ShootingProblem(np.matrix(x0).T, [ model2 ]*T, model2)
    ddp2 = crocoddyl.SolverDDP(problem2)
    done = ddp2.solve()
    print("Without warstart", ddp2.iter)
