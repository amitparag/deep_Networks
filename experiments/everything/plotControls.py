
import numpy as np
import pandas as pd

def plot_statistics(data, index:int = 3, predicted:bool= False):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    sns.set_style("whitegrid")
    mpl.rcParams['figure.dpi'] = 80
    fig = plt.figure(figsize=(8, 6))
    plt.axis(aspect='image')
    
    plt.scatter(data[:, 0], data[:, 1], c=data[:, index], cmap = 'jet', alpha = 0.8, linewidths = 0)
    plt.xlabel("X Coordinates", fontsize = 20)
    plt.ylabel("Y Coordinates", fontsize = 20)
    cb = plt.colorbar()
    if predicted:
        plt.suptitle("Predicted", fontsize = 18, fontweight='bold')
    else:
        plt.suptitle("Crocoddyl", fontsize = 18, fontweight='bold')
        
    if index == 3:
        
        plt.title("Linear velocity", fontsize = 15)
    elif index == 4:
        plt.title("Angular velocity", fontsize = 15)
        
    elif index == 5:
        plt.title("Value Function", fontsize = 15)
        
    else:
        print("...")
    plt.show()    

    
