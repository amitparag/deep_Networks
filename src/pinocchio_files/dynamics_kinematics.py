"""
We show the basics of loading a robot and generating a motion through optmization.

The problem with the normal optimization was the simple cost. In this example, we will add a new cost the cost.
The forward kinematics indeed computes the placement of the last frame, i.e the rotation R and the translation p, denoted M = [R,p] in SE(3).
We need to define a metric to score the distance between to frames M_1 and M_2.
Several metrics can be chosen, but a nice one is given by the SE3 logarithm function,
that converts the gap between two frames into the velocity that should applied (constant)
during t=1 to bridge the gap a displace $M_1$ into M_2.
"""
# Pinocchio is nicely tailored for generating the motion of a robot using a optimization program.
# Look fKinematics_arm.py for a no fuss, no explanation implementation.

import numpy as np
from numpy.linalg import norm, inv, pinv, svd, eig
import robots
import pinocchio
from pinocchio.utils import *
from pinocchio import SE3, Motion, Force, Inertia
from scipy.optimize import fmin_slsqp


# Supress the deprecation warnings
pinocchio.switchToNumpyMatrix()

######################################################################
# Initialising the robot
###########################


# Load the model
robot = robots.loadTalos()
print(robot.model)

# robot.initDisplay(loadModel=True) --> Deprecated
robot.initViewer(loadModel=True)

"""
Gepetto-viewer is indeed a rigid-object viewer, that display each mesh at a given placement (gepetto-viewer has no idea of the kinematic chain).
You then need pinocchio to compute the placement of all the bodies and place them at the right position and orientation.
This is all done in RobotWrapper.
"""

# Display the current configuration of the robot is q0
robot.display(robot.q0)

# Display a random configuration of robot
robot.display(rand(robot.model.nq) * 2 - 1)

rdata = robot.model.createData()
q = rand(robot.model.nq)
pinocchio.forwardKinematics(robot.model, rdata, q)

for i, M in enumerate(rdata.oMi[1:]):
    print(i, M)


# The end effector position is given by
print("End effector = ", rdata.oMi[-1].translation.T)


##########################################################################
# Optimization
######################

"""
Now that er have initialized the robot and learned how to calculate the end effector position for a given configuration, let' optimize.
The problem statement:
        Compute a robot configuration that minimizes the distance between the position of the end-effector and a 3D target.
A caveat:
        Pinocchio has been implemented with the Matrix class. Any other Python package is implemented with the Array class.
        In particular, the SciPy optimizers are with Array.
        So, we will painfully have to convert array to matrix before calling Pinocchio algorithms,
        and back to array when returning the results to the optimizer.
"""

# Step 1. First define a couple of functions to convert matrix into array and vice-versa. Helps keep the code clean.


def matrix_to_array(m): return np.array(m.flat)


def array_to_matrix(a): return np.matrix(a).T


# Step 2. Define a goal
ref = np.matrix([.3, .3, .3]).T

# Step 3. Write a cost function


def cost(x):
    """
    The cost function simply has to call forwardKinematics, and
    return the difference between the computed effector position and a reference.
    """
    q = array_to_matrix(x)

    pinocchio.forwardKinematics(robot.model, rdata, q)

    # The end effector position
    M = rdata.oMi[-1]
    p = M.translation

    residuals = matrix_to_array(p - ref)
    return sum(residuals**2)


# Initial configuration of the robot
x0 = np.random.rand(robot.model.nq)

print("Cost ", cost(x0))

# Optimizer
"""
The optimizer chosen for the class is SLSQP which is a SQP accepting equality, inequality and bound constraints.
It uses BFGS for quasi-newton acceleration and a least-square QP for computing the Newton step.
It is quite a good solver, although not strong enough for implementing real robotics application.
It is yet quite comfortable for a class to have access to it through the easy package SciPy.
"""

# Here we only use the initial guess and the cost function.
"""
result = fmin_slsqp(x0 = np.zeros(robot.model.nq),
                    func = cost,
                    iter = 10)
"""
"""
result = fmin_slsqp(x0=np.random.rand(robot.model.nq),
                    func=cost,
                    iter=10)
q_optimized = array_to_matrix(result)
robot.display(q_optimized)
"""

# Let's use the viewer to see what the solver is doing. First, let's add a visual object to mark the target.
gview = robot.viewer.gui

gview.addSphere('world/target', 0.1, [1., 0., 0., 1.])  # radius, [R,G,B,A]

gview.applyConfiguration(
    'world/target', [.3, .3, .3, 0., 0., 0., 1.])  # x,y,z,quaternion

gview.refresh()


def callbackDisp(x):
    import time

    q = array_to_matrix(x)

    robot.display(q)

    time.sleep(.5)


result = fmin_slsqp(x0=np.zeros(robot.model.nq),
                    func=cost,
                    callback=callbackDisp)
