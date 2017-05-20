from pendulum import *
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

# period_convergency()


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

# draw_freqencies()


def pendulum_path(theta1, theta2):
        g = 9.81
        l = 5
        N = 200
        h = 0.1

        Theta1, Theta2 = double_pendulum(theta1, theta2, g, l, N, h)

        x2 = np.empty(N)
        for i in range(N): x2[i] = np.sin(Theta1[i]) + np.sin(Theta2[i])
        y2 = np.empty(N)
        for i in range(N): y2[i] = -np.cos(Theta1[i]) - np.cos(Theta2[i])

        return x2, y2


def draw_pendulum():

        theta1 = 1
        theta2 = 0
        x2, y2 = pendulum_path(theta1, theta2)
        plt.plot(x2, y2, label="theta1 =" + str(theta1) + ", theta2 = " + str(theta2))

        theta1 = 1
        theta2 = 0.01
        x2, y2 = pendulum_path(theta1, theta2)
        plt.plot(x2, y2, label="theta1 =" + str(theta1) + ", theta2 = " + str(theta2))

        
        plt.legend()
        plt.show()

draw_pendulum()

