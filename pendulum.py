from numpy import empty, arange, array, cos, sin, pi
from cauchy import meth_n_step, step_rk4


# return the abscissia of the intersection point between
# the line ((x1, y1), (x2, y2)) and abscissia axis
def zero(x1, y1, x2, y2):
        a = (y1 - y2) / (x1 - x2)
        b = y1 - (a * x1)
        return -b / a


# return the periode of the function function
# discribe by X and Y
def period(X, Y):
        n = len(X)
        k = 1
        
        while ((k < n) and (Y[k - 1] * Y[k] > 0)) :
                k += 1
        if not(k < n) :
                print("too short: can't find a starting point")
                return 1
        
        # sig(Y[k - 1]) != sig(Y[k])
        t0 = zero(X[k - 1], Y[k - 1], X[k], Y[k])
        nb = 0

        for i in range(k + 1, n) :
                if (Y[i - 1] * Y[i] <= 0) :
                        nb += 1
                        k = i   
        if (nb == 0) :
                print("too short: not a complet periode")
                return 1

        # sig(Y[k - 1]) != sig(Y[k])
        tn = zero(X[k - 1], Y[k - 1], X[k], Y[k])
        T = (tn - t0) / nb # average time of a semi-periode
        return 2 * T

         
# return the frequencies of a simple pendulum
# with initial theta given by the array
def frequencies(Theta, g, l):
        t0 = 0
        h = 0.3
        N = 50 # probably it's enough
        
        x = arange(t0, t0 + N*h, h)
        y = empty(N)
        # theta'' + (g/l)sin(theta) = 0
        F = lambda t, Y: array([Y[1], -(g/l) * sin(Y[0])])

        n = len(Theta)
        freq = empty(n)
        for i in range(n):

                Y0 = array([Theta[i], 0])
                Y = meth_n_step(Y0, t0, N, h, F, step_rk4)
                for j in range(N) :
                        y[j] = Y[j][0]
                        freq[i] = 1 / period(x, y)

        return freq


# Y = [th1, th1', th2, th2']
def step_double_pendulum(Y, g, l):
        th1 = Y[0]
        th11 = Y[1]
        th2 = Y[2]
        th22 = Y[3]
        alpha = 1 / (l * (3 - cos(2 * th1 - 2 * th2)))
        
        th111 = alpha * ((-3 * g * sin(th1))
                         - (g * sin(th1 - 2 * th2))
                         - (2 * sin(th1 - th2) * (th22 * th22 * l
                                                  + th11 * th11 * l * cos(th1 - th2))))

        th222 = alpha * (2 * sin(th1 - th2) * ((th11 * th11 * l * 2)
                                               + (2 * g * cos(th1))
                                               + (th22 * th22 * l * cos(th1 - th2))))

        return array([th11, th111, th22, th222]) # [th1', th1'', th2', th2'']


def pendulum_path(th1, th2, g, l, N):
        h = 0.05
        Y0 = array([th1, 0, th2, 0])
        F = lambda t, Y: step_double_pendulum(Y, g, l)
        sol = meth_n_step(Y0, 0, N, h, F, step_rk4)

        x2 = empty(N)
        y2 = empty(N)
        for i in range(N):
                x2[i] =  sin(sol[i][0]) + sin(sol[i][2])
                y2[i] = -cos(sol[i][0]) - cos(sol[i][2])
                
        return x2, y2


def pendulum_all_info(th1, th2, g, l, N, h):
        Y0 = array([th1, 0, th2, 0])
        F = lambda t, Y: step_double_pendulum(Y, g, l)
        sol = meth_n_step(Y0, 0, N, h, F, step_rk4)

        t1 = empty(N)
        t2 = empty(N)
        x1 = empty(N)
        x2 = empty(N)
        y1 = empty(N)
        y2 = empty(N)
        for i in range(N):
                t1[i] = sol[i][0]
                t2[i] = sol[i][2]
                
                x1[i] = sin(t1[i])
                x2[i] = x1[i] + sin(t2[i])
            
                y1[i] = -cos(t1[i])
                y2[i] = y1[i] - cos(t2[i])
            
        return t1, t2, x1, y1, x2, y2


def flip_over_ratio(th1, th2, g, l):
        N = 300
        h = 0.05
        Y0 = array([th1, 0, th2, 0])
        F = lambda t, Y: step_double_pendulum(Y, g, l)
        sol = meth_n_step(Y0, 0, N, h, F, step_rk4)

        for i in range(N):
                if (sol[i][2] > pi or sol[i][2] < -pi):
                        return i / N
                
        return 1
