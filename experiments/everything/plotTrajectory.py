

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns



def plotTrajectories(cost_trajectory, if_crocoddyl:bool = True):
    """
    Data must be in a dictionary -> key : (x,y)
    
    """
    sns.set_style("whitegrid")
    mpl.rcParams['figure.dpi'] = 80
    fig = plt.figure(figsize=(8, 6))
    
    c = cost_trajectory.keys()
    norm = mpl.colors.Normalize(vmin=min(c), vmax=max(c))
    cmap = mpl.cm.ScalarMappable(norm = norm, cmap=mpl.cm.jet)
    cmap.set_array([])


    for key, trajectory in cost_trajectory.items():
        plt.scatter(trajectory[:, 0], trajectory[:, 1], marker = '', zorder=2, s=50,linewidths=0.2,alpha=.8, cmap = cmap )
        plt.plot(trajectory[:, 0], trajectory[:, 1], c=cmap.to_rgba(key))

    plt.xlabel("X Coordinates", fontsize = 20)
    plt.ylabel("Y Coordinates", fontsize = 20)
    if if_crocoddyl:
        plt.title("Trajectories from Crocoddyl ", fontsize =20)
    else:
        plt.title(" Neural Network Predictions ", fontsize =20)
    plt.colorbar(cmap)
    plt.show()
    
