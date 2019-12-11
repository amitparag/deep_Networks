import random
from neural_net import train_net
import crocoddyl
import numpy as np
crocoddyl.switchToNumpyArray()

class warmstart():
    def __init__(self, n_trajectories: int = 100):
        self.__n_trajectories = n_trajectories
        
    def warm(self):
        net = train_net()
        neuralNet = net.generate_trained_net()
        """
        state_weight: float = 1., # can use random.uniform(1, 3)
                 control_weight: float = 0.3, # can use random.uniform(0,1)
                 nodes: int = 20,
                    
        """  
        for _ in range(self.__n_trajectories):
            
            initial_config = [random.uniform(-2.1, 2.),
                                        random.uniform(-2.1, 2.), 
                                        random.uniform(0, 1)]

            model = crocoddyl.ActionModelUnicycle()
            model2 = crocoddyl.ActionModelUnicycle()
            
            model.costWeights = np.matrix([1., .3]).T
            model2.costWeights = np.matrix([1., .3]).T
    
            
            problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 20, model)
            problem2 = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model2 ] * 20, model2)

            
            ddp = crocoddyl.SolverDDP(problem)
            ddp2 = crocoddyl.SolverDDP(problem2)
            
            
            x_data = np.matrix(initial_config).reshape(1, 3)

            prediction = neuralNet.predict(x_data).reshape(20, 5)

            xs = np.matrix(prediction[:, 0:3])

            us = np.matrix(prediction[:, 3:5])

            ddp_xs = []
            ddp_xs.append(np.matrix(initial_config).reshape(1, 3).T)
            for _ in  range(xs.shape[0]):
                ddp_xs.append((np.matrix(xs[_]).T))

            ddp_us = []
            for _ in  range(us.shape[0]):
                ddp_us.append(np.matrix(us[_]).T)

            ddp.solve([], ddp_us)
            ddp2.solve()
            print(f" With warmstart, without warmstart ")
            print(ddp.iter, "  ", ddp2.iter)
            
            
            
if __name__=='__main__':
    warmstart = warmstart()
    warmstart.warm()
