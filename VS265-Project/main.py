import pickle, signal, sys, util
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
	if sys.argv[1] == "estimate"
		estimate(sys.args)
	elif sys.argv[1] == "generate":
		generate_data(sys.argv)
	else:
		print("Input not as expected:", sys.argv[1])
		sys.exit(-1)