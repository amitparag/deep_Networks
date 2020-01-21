from math import cos, sin
import random
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def animateCartpole(xs, sleep=50):
    print("processing the animation ... ")
    cart_size = 1.
    pole_length = 5.
    fig = plt.figure()
    ax = plt.axes(xlim=(-8, 8), ylim=(-6, 6))
    patch = plt.Rectangle((0., 0.), cart_size, cart_size, fc='b')
    line, = ax.plot([], [], 'k-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        ax.add_patch(patch)
        line.set_data([], [])
        time_text.set_text('')
        return patch, line, time_text

    def animate(i):
        x_cart = np.asscalar(xs[i][0])
        y_cart = 0.
        theta = np.asscalar(xs[i][1])
        patch.set_xy([x_cart - cart_size / 2, y_cart - cart_size / 2])
        x_pole = np.cumsum([x_cart, -pole_length * sin(theta)])
        y_pole = np.cumsum([y_cart, pole_length * cos(theta)])
        line.set_data(x_pole, y_pole)
        time = i * sleep / 1000.
        time_text.set_text('time = %.1f sec' % time)
        return patch, line, time_text

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=sleep, blit=True)
    print("... processing done")
    plt.show()
    return anim


import crocoddyl
import numpy as np
import example_robot_data
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum


crocoddyl.switchToNumpyMatrix()


def cartpole():
    # Loading the double pendulum model
    robot = example_robot_data.loadDoublePendulum()
    robot_model = robot.model

    state = crocoddyl.StateMultibody(robot_model)
    actModel = ActuationModelDoublePendulum(state, actLink=1)

    weights = np.array([1, 1, 1, 1] + [0.1] * 2)
    runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
    terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)
    xRegCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuad(state.ndx), state.zero(), actModel.nu)
    uRegCost = crocoddyl.CostModelControl(state, crocoddyl.ActivationModelQuad(1), actModel.nu)
    xPendCost = CostModelDoublePendulum(state, crocoddyl.ActivationModelWeightedQuad(np.matrix(weights).T), actModel.nu)

    dt = 1e-2

    runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
    runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)
    terminalCostModel.addCost("xGoal", xPendCost, 1e4)

    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, runningCostModel), dt)
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, terminalCostModel), dt)
    
    return runningModel, terminalModel


def plot_statistics(df, index:int = 5):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    sns.set_style("whitegrid")
    mpl.rcParams['figure.dpi'] = 80
    fig = plt.figure(figsize=(12, 10))
    plt.axis(aspect='image')
    
    data = df.values
    plt.scatter(data[:, 0], data[:, 1], c=data[:, index], cmap = 'jet', alpha = 0.8, linewidths = 0)
    plt.xlabel("X Coordinates", fontsize = 20)
    plt.ylabel("Y Coordinates", fontsize = 20)
    cb = plt.colorbar()
    if index == 4:
        plt.title("policy", fontsize = 20)
        plt.savefig("policy.png")
        
    elif index == 5:
        plt.title("Value Function", fontsize = 20)
        plt.savefig("value.png")

    elif index == 6:
        plt.title("Iterations", fontsize = 20)
        plt.savefig("iterations.png")

        
    else:
        print("...")
    plt.show()    

import time, multiprocessing, pandas as pd

def get_data(ntraj: int = 1):
    data = []
    T = 100

    for _ in range(ntraj):
        x0 = [ random.uniform(0,1), random.uniform(-3.14, 3.14), random.uniform(0., 0.99)
                    , random.uniform(0., .99)]
        runningModel, terminalModel = cartpole()



        problem = crocoddyl.ShootingProblem(np.matrix(x0).T, [runningModel] * T, terminalModel)
        ddp = crocoddyl.SolverFDDP(problem)
        ddp.solve()
        if ddp.iter < 1000:
            position = []
            # Attach state four vectors
            position.extend(float(i) for i in ddp.xs[0])

            # Attach control
            position.extend(float(i) for i in ddp.us[0])
            # Attach cost
            position.append(sum(d.cost for d in ddp.datas()))
            # Attach the number of iterations
            position.append(ddp.iter)
            data.append(position)

    data = np.array(data)
    
    #df = pd.DataFrame(data[0:,0:], columns = ["x_position", "y_position",
    #                                           "z_position","xs_4",
     #                                         "control", "value_function",
    #                                          "iterations"])
    #plot_statistics(df, index = 4)
    #plot_statistics(df, index = 5)
    
if __name__ == "__main__":

    starttime = time.time()

    get_data(100)

    print('That took {} seconds'.format(time.time() - starttime))


