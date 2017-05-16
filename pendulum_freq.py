import numpy as np
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
        
        x = np.arange(t0, t0 + N*h, h)
        y = np.empty(N)
        # theta'' + (g/l)sin(theta) = 0
        F = lambda t, Y: np.array([Y[1], -(g/l) * np.sin(Y[0])])

        n = len(Theta)
        freq = np.empty(n)
        for i in range(n):

                Y0 = np.array([Theta[i], 0])
                Y = meth_n_step(Y0, t0, N, h, F, step_rk4)
                for j in range(N) :
                        y[j] = Y[j][0]

                freq[i] = 1 / period(x, y)

        return freq
