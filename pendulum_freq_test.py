from pendulum_freq import *
import numpy as np
import matplotlib.pyplot as plt


def test_period():

        expected = 2 * np.pi
        nb_points = 15
        x = np.empty(nb_points)
        y = np.empty(nb_points)
        
        for i in range(nb_points):
                X = np.arange(0, 10, 10 / (i + 6))
                Y = np.sin(X)
                
                x[i] = len(X)
                y[i] = abs(expected - period(X, Y))

        plt.title("Erreur relative de periode")
        plt.xlabel("nombre de point")
        # plt.yscale("log")
        plt.plot(x, y)
        plt.show()

test_period()


