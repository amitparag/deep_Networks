import pandas as pd
import numpy as np
import crocoddyl
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style("whitegrid")
mpl.rcParams['figure.dpi'] = 80



plt.axis(aspect='image')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.1, 2.1)
crocoddyl.switchToNumpyArray()



def plot_trajectories():
    df = pd.read_csv("warmstart.csv")
    data = df.values
    row = data[0,:]
    
    init_xs = []
    init_us = []
    starting_state = row[0:3]
    init_xs.append(np.array(starting_state))

    full_state = np.array(row[3:])
    full_state = full_state.reshape(30, 6)
    state = full_state[:,0:3]
    control = full_state[:,3:5]
    assert state.shape[0] == control.shape[0]
    assert state.shape[1]  == 3
    assert control.shape[1] == 2
    for state in state:
        init_xs.append(np.array(state))
    for control in control:
        init_us.append(np.array(control))




#...... WARMSTARTING CROCODDYL
    model = crocoddyl.ActionModelUnicycle()
    model.costWeights = np.matrix([1, 0.3]).T
    problem = crocoddyl.ShootingProblem(np.matrix(starting_state).T, [ model ] * 30, model)
    ddp = crocoddyl.SolverDDP(problem)
    ddp.solve(init_xs, [], 1000)
    ddp_xs = np.array(ddp.xs)

#..... COLDSTARTING CROCODDYL    
    model2 = crocoddyl.ActionModelUnicycle()
    model2.costWeights = np.matrix([1, 0.3]).T
    problem2 = crocoddyl.ShootingProblem(np.matrix(starting_state).T, [ model2 ] * 30, model2)
    ddp2 = crocoddyl.SolverDDP(problem2)
    ddp2.solve([], [], 1000)
    ddp2_xs = np.array(ddp2.xs)
    print("PLotting 2 trajectories for comparision \n")

    print("The value functions, iterations for warmstarted and coldstarted trajectories are as follows ","Cost :", ddp.cost,",", ddp2.cost, "\n", "iterations ",ddp.iter,",", ddp2.iter )

    
    
    fig=plt.figure(figsize=(20,10))
    ax=fig.add_subplot(111)
    ax.set_xlim(xmin=-2.1, xmax=2.1)
    ax.set_ylim(ymin=-2.1, ymax=2.1)
    ax.plot(ddp_xs[:,0], ddp_xs[:,1], c = 'red', label = 'Warmstarted crocoddyl.')
    ax.scatter(ddp2_xs[:,0], ddp2_xs[:,1], c = 'green', label = 'Coldstarted crocoddyl')
    plt.legend()
    plt.show()
