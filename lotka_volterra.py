from cauchy_test import *
import numpy as np

##Malthus

##Insee Mars 2017 France Metropolitaine
b = 10.9/1000
d = 8.7/1000
y0 = np.array([64862000])

t0 = 2017
N = 1000
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
a = 2.0
b = 4.0
c = 1.0
d = 1.0
f = lambda t,y : np.array([y[0]*(a-b*y[1]), y[1]*(c*y[0]-d)])
y0 = np.array([20.0, 20.0])
t0 = 0.0
N = 100
h = 1.0
graph_2d(y0,t0,N,h,f,0)
