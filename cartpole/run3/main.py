if __name__=="__main__":

    import pandas as pd
    import random
    import numpy as np


    df = pd.read_csv("fooBar.csv")
    data = df.values

    data.shape

    print("Shape x_train -> 0 : 4, value function = index 4, then (x, y, z, theta, control, cost)..repeated")
    print("T -> 50, so T x 6 = 300, + 5 ")

    from baselineNet import *

    x_data = data[:,0:4]
    y_data = data[:,4:]

    print(x_data.shape)
    print(y_data.shape)

    net = kerasNet(x_data,
                 y_data,
                 NUNITS_INPUT = 32,
                 NUNITS = 16,
                 NHIDDEN = 2,
                 lr = 1e-3,
                 EPOCHS = 100,
                 BATCHSIZE = 64,
                 validation_split = 0.1,
                 VERBOSE = 2,
                 optimizer = "adam",
                 loss = ['mean_squared_error'],
                 use_gpu = True,
                 saveModel = False,
                 plot_results = True,
                 baseline = False    )

    from timeit import default_timer as timer
    import crocoddyl
    from data import *
    from plotTable import *
    crocoddyl.switchToNumpyArray()


    iterations = []
    time = []
    for _ in range(100):


        x0 = [random.uniform(0,1), random.uniform(-3.14, 3.14), random.uniform(0., 1), 0.]
        x_test = np.array(x0).reshape(1, -1)
        y_pred_ = net.predict(x_test)

        y_pred = y_pred_[:,1:]
        warm = y_pred.reshape(50, 6)
        init_xs = []
        init_us = []
        init_xs.append(np.array(x0))
        state_array = warm[:,0:4]
        control_array = warm[:,4]

        for state in state_array:
            state = np.matrix(state).T
            init_xs.append(state)

        for control in control_array:
            control = np.matrix(control).T
            init_us.append(control)    





        cartpoleDAM = DifferentialActionModelCartpole()
        cartpoleData = cartpoleDAM.createData()
        x = cartpoleDAM.state.rand()
        u = np.zeros(1)
        cartpoleDAM.calc(cartpoleData, x, u)
        cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True) 
        timeStep = 5e-2
        cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)
        T  = 50
        problem = crocoddyl.ShootingProblem(np.array(x0).T, [ cartpoleIAM ]*T, cartpoleIAM)
        ddp = crocoddyl.SolverDDP(problem)



        cartpoleDAM2 = DifferentialActionModelCartpole()
        cartpoleData2 = cartpoleDAM2.createData()
        x2 = cartpoleDAM2.state.rand()
        u2 = np.zeros(1)
        cartpoleDAM2.calc(cartpoleData2, x2, u2)
        cartpoleND2 = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM2, True) 
        timeStep = 5e-2
        cartpoleIAM2 = crocoddyl.IntegratedActionModelEuler(cartpoleND2, timeStep)
        T  = 50
        problem2 = crocoddyl.ShootingProblem(np.array(x0).T, [ cartpoleIAM2 ]*T, cartpoleIAM2)
        ddp2 = crocoddyl.SolverDDP(problem2)

        start1 = timer()
        ddp.solve(init_xs, init_us, 1000)
        end1 = timer()

        start2 = timer()
        ddp2.solve([], [], 1000)
        end2 = timer()



        ddp_xs = np.array(ddp.xs)
        ddp_xs2 = np.array(ddp2.xs)

        iterations.append(np.array([ddp.iter, ddp2.iter]))
        time.append(np.array([end1- start1, end2 - start2]))

    plotTable(iterations)

    from plot1d import *

    iterations = np.array(iterations)
    time = np.array(time)
    plot_1d(iterations[:,0], xlabel = "Warmstarted" ,ylabel="iterations")
    plot_1d(iterations[:,1], xlabel = "oldstarted" ,ylabel="iterations")
    plot_1d(time[:,0])
    plot_1d(time[:,1])







