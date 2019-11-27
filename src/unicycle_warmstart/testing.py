

from keras.models import load_model
import random
from neural_net import train_net
import crocoddyl
import numpy as np
random.seed(1)
neuralNet = load_model('model.h5')


initial_config = np.matrix([random.uniform(-2.1, 2.),
                            random.uniform(-2.1, 2.), 
                            random.uniform(0, 1)]).T

#print(initial_config.shape)


model = crocoddyl.ActionModelUnicycle()
model.costWeights = np.matrix([1., .3]).T
problem = crocoddyl.ShootingProblem(initial_config, [ model ] * 20, model)
ddp = crocoddyl.SolverDDP(problem)
x_data = re = np.asarray(initial_config.reshape(1, 3))
prediction = neuralNet.predict(x_data).reshape(20, 5)
xs = np.array(prediction[:, 0:3])
us = np.array(prediction[:, 3:5])

ddp_xs = []
ddp_xs.append((initial_config.flatten()))
for _ in  range(xs.shape[0]):
    ddp_xs.append(np.array(xs[_]))

ddp_us = []
for _ in  range(us.shape[0]):
    print(us[_])
    ddp_us.append(np.array(us[_]))
    

ddp.solve(ddp_xs, ddp_us)
#print(ddp.iter)
#print(ddp_xs)
print(type(ddp_xs))
print(type(ddp_us))
