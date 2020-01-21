
import crocoddyl
import numpy as np
import example_robot_data
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
crocoddyl.switchToNumpyMatrix()


#%%file cartpole.py
from math import cos, sin
import random
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
