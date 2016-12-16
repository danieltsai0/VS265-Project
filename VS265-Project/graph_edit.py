import array, sys, os, math, config, pickle
import numpy as np
import matplotlib.pyplot as plt

def graph_edit(d, nnp, Tnum):
	path = config.gen_data_dir + str(d) + "_" + str(nnp) + "_" + str(Tnum) + config.entropy_pickle_suffix
	with open(path, "rb") as f:
		g = pickle.load(f)
	xs = []
	ys = []
	for pt in g:
		xs.append(pt[0])
		ys.append(pt[1])
	fig, ax = plt.subplots()
	ax.set_xscale('log', basex=2)
	ax.plot(xs, ys)
	plt.xlabel("Number of samples")
	plt.ylabel("Entropy (bits/pixel)")
	plt.title("Entropy vs. Number of Samples for " + str(d) + " data, N: 2^" + str(nnp) + ", T: " + str(Tnum))
	plt.savefig(d + "_" + str(nnp) + "_" + str(Tnum) + "_entropy_plot.png")

graph_edit("pca", 17, 10000)