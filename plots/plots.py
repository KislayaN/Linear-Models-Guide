import matplotlib.pyplot as plt
import seaborn as sns


class Plots:
    def __init__(self):
        pass
    
    def scatter_plot(self, X_axis, y_axis, xlabel, ylabel, title):
        plt.scatter(
            x=X_axis,
            y=y_axis,
            alpha=0.5
            )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()