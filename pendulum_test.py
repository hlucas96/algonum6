from pendulum import *
import numpy as np
import matplotlib.pyplot as plt


def test_periode():
	X = np.arange(0, 10, 0.4)

	Y = np.sin(X)
	T1 = periode(X, Y)

	Y = np.cos(X)
	T2 = periode(X, Y)

	plt.plot(X, Y, "o-")
	plt.show()

	print(T1, T2, 2 * np.pi)

test_periode()
