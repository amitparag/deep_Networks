import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def plot_1d(l, colors=None, xlabel='', ylabel=''):
    """Plot a 1D list l of numbers."""
    ax = sns.barplot([i for i in range(len(l))],
                     l,
                     palette=colors)
    ax.set(xlabel=xlabel, ylabel=ylabel, label='big')
    ax.set_xticks([])
    plt.show()

