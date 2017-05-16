from pendulum_freq import *
import numpy as np
import matplotlib.pyplot as plt


def period_convergency():
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

period_convergency()


def draw_freqencies():
        g = 9.81
        l = 2
        f = np.sqrt(g / l) / (2 * np.pi)
        h = 0.1
        theta = np.arange(-np.pi + h, np.pi, h)
        freq = frequencies(theta, g, l)

        plt.title("Frequences d'oscillations d'un pendule simple")
        plt.plot(theta, freq)
        plt.plot([-np.pi, np.pi], [f, f], label="sqrt(g / L) / 2pi")
        plt.xlabel("angle initial")
        plt.legend()
        plt.show()

draw_freqencies()

