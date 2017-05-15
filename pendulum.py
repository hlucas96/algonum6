import numpy as np

g = 9.81
l = 5



# return the intersection point between
# the line ((x1, y1) (x2, y2)) and abscissia axis
def zero(x1, y1, x2, y2):
	
	a = (y1 - y2) / (x1 - x2)
	b = y1 - (a * x1)

	return -b / a
	

# return the periode of the function function
# discribe by X and Y
def periode(X, Y):
	
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
	l = k
	
	for i in range(k + 1, n) :

		if (Y[i - 1] * Y[i] <= 0) :
			nb += 1
			l = i
			
	if (nb == 0) :
		print("too short: not a complet periode")
		return 1


	tn = zero(X[l - 1], Y[l - 1], X[l], Y[l])
	T = (tn - t0) / nb # average time of a semi-periode

	return 2 * T
