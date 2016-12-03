import time
import numpy as np

def func(a, ln):
	for n in ln:
		z = a[:n]

def func2(a, ln):
	z = []
	prevn = 0
	for n in ln:
		z.extend(a[prevn:n])
		prevn = n

# def func3(a, ln):
# 	for n in ln:
# 		z = a[:n]

k = 1000000
a = np.random.rand(k)
ln = np.power(2, range(1,20))

s1 = time.time()
func(a, ln)
print("runtime: ",time.time()-s1)

s2 = time.time()
func2(a, ln)
print("runtime: ",time.time()-s2)