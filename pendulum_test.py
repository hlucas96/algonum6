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
        plt.ylabel("Hz")
        plt.legend()
        plt.show()


def double_pendulum(th1, th2, g, l, time):
        h = 0.05
        N = int(time / h)
        t1, t2, x1, y1, x2, y2 = pendulum_all_info(th1, th2, g, l, N, h)

        fig, ax = plt.subplots()
        ax.set_xlim([-2 * l - h, 2 * l + h])
        ax.set_ylim([-2 * l - h, 2 * l + h])
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)

        def init():
                line.set_data([], [])
                return line,

        def animate(i):
                line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
                return line,

        ani = animation.FuncAnimation(fig, animate, frames=N, interval=1000*h, blit=True, init_func=init, repeat=True)
        plt.show()


def draw_pendulum():
        N = 300
        g = 9.81
        l = 5

        plt.subplot(1, 3, 1)
        #plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        th1 = 2
        th2 = 2
        plt.title("th1: " + str(th1) + " rad\n th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, N)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")
        
        plt.subplot(1, 3, 2)
        #plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        th1 = th1
        th2 = th2 + 0.1
        plt.title("th1: " + str(th1) + " rad\n th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, N)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")

        plt.subplot(1, 3, 3)
        #plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        th1 = th1
        th2 = th2 + 0.1
        plt.title("th1: " + str(th1) + " rad \n th2: " + str(th2) + "rad")
        x, y = pendulum_path(th1, th2, g, l, N)
        plt.axis("equal")
        plt.plot(x, y)
        linex = [0,  np.sin(th1),  np.sin(th1) + np.sin(th2)]
        liney = [0, -np.cos(th1), -np.cos(th1) - np.cos(th2)]
        plt.plot(linex, liney, "o-")

        plt.show()


def draw_flip_over(N):
        g = 9.81
        l = 5
        angle = lambda i: -3 + (i * 6 / N)
        
        M = np.empty((N, N))
        for i in range(N):
                for j in range(N):
                        M[i][j] = flip_over_ratio(angle(j), angle(i), g, l)

        fig = plt.figure(5)
        ax = plt.subplot(111)
        im = ax.imshow(M, cmap=plt.get_cmap("cubehelix"))
        fig.colorbar(im)

        # fig.savefig("figure.png")
        plt.show()


        
if __name__ == "__main__":
        
        period_convergency()
        draw_freqencies()
        double_pendulum(3, -3, 9.81, 1, 30)
        draw_pendulum()
        
        print("result in 10sec for a 20 x 20 image")
        draw_flip_over(20)
