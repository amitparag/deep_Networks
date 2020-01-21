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
    
    
def cartpole_data(ntraj:int = 1000):
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
        cost = []    

        x0 = np.array([ random.uniform(0,1), random.uniform(-3.14, 3.14), random.uniform(0., 1), 0. ])

        problem = crocoddyl.ShootingProblem(x0.T, [ cartpoleIAM ]*T, cartpoleIAM)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000)
        if ddp.iter < 1000:
            xs = np.array(ddp.xs)
            us = np.array(ddp.us)
            
            for d in ddp.datas():
                cost.append(d.cost)
            
            
        data.append(xs.ravel())   
        
    data = np.array(data)
    
    
    return data

#x_data, y_data = cartpole_data(5000)
if __name__=="__main__":
    
    from multiprocessing import Pool
    import time

    start = time.time()
    p = Pool(processes=20)
    data = p.map(cartpole_data, (10000,))
    p.close()
    end = time.time()
    print("time ", start -end)

    data = np.squeeze(np.array(data))
    numpy.savetxt("fooBar.csv", data, delimiter=",")
