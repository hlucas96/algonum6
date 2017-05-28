from cauchy_test import *
import numpy as np
import matplotlib.pylab as plt

##Malthus

##Insee Mars 2017 France Metropolitaine
b = 10.9/1000
d = 8.7/1000
y0 = np.array([64862000])

t0 = 2017
N = 10000
h = 0.1
f = lambda t,y : (b-d)*y[0]
graph_N_step(y0,t0,N,h,f,0)


#Verhulst
gamma = 0.1
kappa = 80
f = lambda t,y : gamma*y[0]*(1-y[0]/kappa)
y0 = np.array([60])
t0 = 2017
N = 1000
h = 0.1
graph_N_step(y0,t0,N,h,f,0)

##Lotka-Volterra
a = 2.0/3.0
b = 4.0/3.0
c = 1
d = 1
f = lambda t,y : np.array([y[0]*(a-b*y[1]), y[1]*(c*y[0]-d)])
y0 = np.array([1.0, 1.0])
t0 = 0.0
N = 30000
h = 1e-3
graph_2d_pred(y0,t0,N,h,f)

def periode(y0, t0, N, h, f, meth):
    Y = np.empty(N+1, dtype=object)
    Y[0] = y0
    for i in range(N):
        Y[i+1] = meth(Y[i], t0 + i*h, h, f)
        if (Y[i][0] > y0[0] and Y[i+1][0] <= y0[0]):
            return t0 + (i+1)*h
print("Periode des fonctions : ",periode(y0, t0, N, h, f, step_rk4))


def graph_P_N(y0, t0, N, h, f, meth, nb_courbes, pas):
    for i in range(nb_courbes):
        x, y = extract2(meth_n_step(y0, t0, N, h, f, meth))
        plt.plot(x, y, "k")
        y0[0] += pas
    plt.xlabel("Proies")
    plt.ylabel("Predateurs")
    plt.show()
y0 = np.array([1.0, 0.5])
N = 10000
graph_P_N(y0,t0,N,h,f,step_rk4, 10, 0.1)
