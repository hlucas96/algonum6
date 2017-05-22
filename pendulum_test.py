from pendulum import *
from matplotlib import animation
from matplotlib import pyplot as plt
import numpy as np

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


# fig = plt.figure()

# ax = plt.axes(xlim=(0, 2), ylim=(0, 100))

# N = 4
# lines = [plt.plot([], [])[0] for _ in range(N)]

# def init():    
#     for line in lines:
#         line.set_data([], [])
#     return lines

# def animate(i):
#     for j,line in enumerate(lines):
#         line.set_data([0, 2], [10 * j,i])
#     return lines

# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=100, interval=20, blit=True)

# plt.show()


def draw_pendulum():
        N = 300
        h = 0.1
        m = 1
        g = 9.81
        l = 5

        plt.subplot(1, 3, 1)
        #plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        th1 = 2
        th2 = 1.5
        plt.title("th1: " + str(th1) + "rad , th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, m, N, h)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")
        
        plt.subplot(1, 3, 2)
        #plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        th1 = th1 + 0.02
        th2 = th2
        plt.title("th1: " + str(th1) + "rad , th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, m, N, h)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")

        plt.subplot(1, 3, 3)
        #plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        th1 = th1 + 0.02
        th2 = th2
        plt.title("th1: " + str(th1) + "rad , th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, m, N, h)
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
        m = 1
        angle = lambda i: -3 + (i * 6 / N)
        
        M = np.empty((N, N))
        for i in range(N):
                for j in range(N):
                        M[i][j] = flip_over_ratio(angle(j), angle(i), g, l, m)

        fig = plt.figure(5)
        ax = plt.subplot(111)
        im = ax.imshow(M, cmap=plt.get_cmap("YlGn"))
        fig.colorbar(im)
        plt.show()

draw_flip_over(20)
