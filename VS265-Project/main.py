import pickle, signal, sys, util
import numpy as np
import matplotlib.pyplot as plt

from util import *
from generate_data import *
from estimate import *



if __name__ == "__main__":
	if sys.argv[1] == "estimate"
		estimate.estimate(sys.args)
	elif sys.argv[1] == "generate":
		generate_data.generate_data(sys.argv)
	else:
		print("Input not as expected:", sys.argv[1])
		sys.exit(-1)