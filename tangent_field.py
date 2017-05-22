import matplotlib.pyplot as plt
import numpy as np

from cauchy import *

# Given the solution of the equadiff, computes its tangent field
def compute_tangent_field(T, h) :
    R = []
    for i in range(1, len(T) - 1) :
        R.append([
            (T[i-1][0]-T[i+1][0])/(2*h),
            (T[i-1][1]-T[i+1][1])/(2*h)
            ])
    return R

# T is a matrix. T[n] is the value (2-dimension array) of the function at time h*n
def display_tangent_field(T, h) :
    R = compute_tangent_field(T, h)
    T = T[1:len(T)-1]
    X=[]
    Y=[]
    U=[]
    V=[]
    for i in range(len(T)) :
        X.append(T[i][0])
        Y.append(T[i][1])
        U.append(R[i][0])
        V.append(R[i][1])
    plt.figure()
    Q = plt.quiver(X, Y, U, V, units='xy', width=0.3)
    plt.show()

def tangent_field(y0, t0, N, h, f, meth) :
    display_tangent_field(
        meth_n_step(y0, t0, N, h, f, meth), h)
    
def tests() :
    tangent_field(np.array([1, 0]), 0, 60, 0.4,
                  lambda t,Y : np.array([-Y[1], Y[0]]),
                  step_euler)

if __name__ == "__main__" :
    tests()
