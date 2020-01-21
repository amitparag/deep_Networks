import random
import crocoddyl
import pinocchio
import numpy as np
crocoddyl.switchToNumpyArray()

class DifferentialActionModelCartpole(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(4), 1, 6) # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)

        self.m1 = 1.
        self.m2 = .1
        self.l  = .5
        self.g  = 9.81
        self.costWeights = [ 1., 1., 0.1, 0.001, 0.001, 1. ]  # sin, 1-cos, x, xdot, thdot, f
        
    def calc(self, data, x, u=None):
        if u is None: u=model.unone
        # Getting the state and control variables
        y, th, ydot, thdot = np.asscalar(x[0]), np.asscalar(x[1]), np.asscalar(x[2]), np.asscalar(x[3])
        f = np.asscalar(u[0])

        # Shortname for system parameters
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        m = m1 + m2
        mu = m1 + m2 * s**2
        xddot  = (f     + m2 * c * s * g - m2 * l * s * thdot**2 ) / mu
        thddot = (c * f / l + m * g * s / l  - m2 * c * s * thdot**2 ) / mu
        data.xout = np.matrix([ xddot,thddot ]).T

        # Computing the cost residual and value
        data.r = np.matrix(self.costWeights * np.array([ s, 1-c, y, ydot, thdot, f ])).T
        data.cost = .5* np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self,data,x,u=None,recalc=True):
        # Advance user might implement the derivatives
        pass
    
    
def cartpole_data(ntraj:int = 1):
    cartpoleDAM = DifferentialActionModelCartpole()
    cartpoleData = cartpoleDAM.createData()
    x = cartpoleDAM.state.rand()
    u = np.zeros(1)
    cartpoleDAM.calc(cartpoleData, x, u)
    cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True) 
    timeStep = 5e-2
    cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)
    T  = 50
    
    
    data = []
    
    for _ in range(ntraj):

        x0 = [ random.uniform(0,1), random.uniform(-3.14, 3.14), random.uniform(0., 1), 0.]

        problem = crocoddyl.ShootingProblem(np.array(x0).T, [ cartpoleIAM ]*T, cartpoleIAM)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        if ddp.iter < 1000:
            ddp_xs = np.array(ddp.xs)
            # Append value function to initial state
            x0.append(ddp.cost)
            
            # get us, xs for T timesteps
            us = np.array(ddp.us)
            
            xs = ddp_xs[1:,:]

            cost = []
            for d in ddp.datas():
                cost.append(d.cost)
            # Remove the first value    
            cost.pop(0)    
            cost = np.array(cost).reshape(-1, 1)
            info = np.hstack((xs, us, cost))
            info = info.ravel()
            state = np.concatenate((x0, info))
            
        data.append(state)
            
            
        
        
    data = np.array(data)
    
    print("Shape x_train -> 0 : 4, value function = index 4, then (x, y, z, theta, control, cost)..repeated")
    print("T -> 50, so T x 6 = 300, + 5 ")
    
    #np.savetxt("foo.csv", data, delimiter=",")
    return data


if __name__ == "__main__":
    import numpy as np
    import multiprocessing as mp
    pool = mp.Pool(processes=8)
    results = [pool.apply_async(cartpole_data, args=(10000,))]
    data = [p.get() for p in results]    
    data = np.squeeze(np.array(data))
    np.savetxt("fooBar.csv", data, delimiter=",")
    print(data.shape)
