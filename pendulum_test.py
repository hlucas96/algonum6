from pendulum import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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


def draw_pendulum():
        N = 300
        h = 0.1
        g = 9.81
        l = 5

        # TODO, fix limit of frame

        plt.subplot(1, 3, 1)
        th1 = 2
        th2 = 1.5
        plt.title("th1: " + str(th1) + "rad , th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, N, h)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")
        
        plt.subplot(1, 3, 2)
        th1 = th1 + 0.02
        th2 = th2
        plt.title("th1: " + str(th1) + "rad , th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, N, h)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")

        plt.subplot(1, 3, 3)
        th1 = th1 + 0.02
        th2 = th2
        plt.title("th1: " + str(th1) + "rad , th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, N, h)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")

        plt.show()

# draw_pendulum()


def draw_flip_over(N):
        g = 9.81
        l = 5
        angle = lambda i: -3 + (i * 6 / N)
        
        M = np.empty((N, N))
        for i in range(N):
                for j in range(N):
                        M[i][j] = flip_over_ratio(angle(i), angle(j), g, l)

        fig = plt.figure(5)
        ax = plt.subplot(111)
        im = ax.imshow(M, cmap=plt.get_cmap('hot'))
        fig.colorbar(im)
        plt.show()

draw_flip_over(300)
