"""
x and y are randomly sampled while theta(z) = 0

"""
import random, numpy as np, crocoddyl

from mpl_toolkits.mplot3d import axes3d


from matplotlib import cm

import matplotlib.pyplot as plt
crocoddyl.switchToNumpyArray()

fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X Positions")
ax.set_ylabel("Y Positions")
ax.set_zlabel("Velocity")
ax.set_xlim([-2., 2.])
ax.set_ylim([-2., 2])
ax.text(0, 0, 0, "Goal", color='black')


for _ in range(10):
    starting_configurations, optimal_trajectories, controls = [], [], []
    initial_config = [random.uniform(-2.1, 2.1), 1, 1]
    model = crocoddyl.ActionModelUnicycle()

    model.costWeights = np.matrix([1, 0.3]).T

    problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 20, model)
    ddp = crocoddyl.SolverDDP(problem)
    ddp.solve([], [], 100)
    state = np.squeeze(np.array(ddp.xs))
    control = np.squeeze(np.array(ddp.us))

    x = state[1:,0]
    y  =state[1:,1]
    theta = state[1:,2]
    z = velocity = control[:,0]
    force = control[:,1]
    ax.plot3D(x, y, z)
    ax.set_title("y = 1, theta = 1")

plt.show()
