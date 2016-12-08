import pickle, signal, sys, util, config, sys, time
import numpy as np
import matplotlib.pyplot as plt
import random

"""
Estimates entropy and relative dimension of a given dataset. 

Arguments: 
	n_neighbors_pow: used to exponentiate the number dos
	tod: type(s) of data being measured 

Example calls:
	python main.py estimate 17 nat ica
	python main.py estimate 17 nat ica pca

	Note that the order of the tod doesn't matter
"""
vlog = np.vectorize(util.log)
def estimate(*args):

	# try:
		_, _, n_neighbors_pow, *tod = args
		estimates, nnp = [], int(n_neighbors_pow)

		for d in tod:

			if d not in config.types_of_data:
				print("Unknown type of data requested. Skipping entropy calculation for", d)
				continue

			# Load patches 
			Tp, Nps = util.load_patches(d, nnp)
			if d == "gwn": #87723
				Tp = Tp[np.random.choice(len(Tp), 50)]
			elif d == "nat": #581405
				Tp = Tp[np.random.choice(len(Tp), 50)]
			print("T:", len(Tp))
			print("N:", len(Nps))
			# Generate distances for each target patch
			i = 0
			k = len(Tp[0])**2
			xs, ys = [], []
			print(Tp[0])
			while i < nnp: 
				# print(time.time())
				Npcurr = Nps[np.random.choice(len(Nps), 2**i)]
				Dstars = [util.find_nearest_neighbor(patch, Npcurr) for patch in Tp]
				Dstar = (1/len(Tp))*np.sum(vlog(Dstars))
				print(Dstar)
				entropy = util.entropy_est(k, Dstar, 2**i)
				entropy = entropy/k #need entropy per pixel
				print(2**i, entropy)
				xs.append(2**i)
				ys.append(entropy)
				estimates.append([2**i, entropy])
				i+=1
			fig, ax = plt.subplots()
			ax.set_xscale('log', basex=2)
			ax.plot(xs, ys)
			plt.show()

			plt.savefig(d + "_entropy_plot.png")
			plt.close()
			

		pickle.dump(estimates, open(config.gen_data_dir 
							   + n_neighbors_pow
							   + config.entropy_pickle_suffix, "wb"))

	# except ValueError:
	# 	print("incorrect command line argument formatting", args)
	# 	sys.exit(-1)


# gwnims = util.load_dataset("gwn")
# print(len(gwnims))
# util.view(gwnims[0])

# Tp, Np = util.load_patches("gwn", 18)
# print(len(Tp), len(Np))


def avg_nn_dist(*args):
	_, _, n_neighbors_pow, *tod = args
	estimates, nnp = [], int(n_neighbors_pow)

	for d in tod:

		if d not in config.types_of_data:
			print("Unknown type of data requested. Skipping entropy calculation for", d)
			continue

		# Load patches 
		Tp, Nps = util.load_patches(d, nnp)
		if d == "gwn":
			Tp = Tp[np.random.choice(len(Tp), 87723)]
		elif d == "nat":
			Tp = Tp[np.random.choice(len(Tp), 581405)]
		print("T:", len(Tp))
		print("N:", len(Nps))
		# Generate distances for each target patch
		i = 0
		k = len(Tp[0])**2 #patches still in matrix form
		print(k)
		xs, ys = [], []
		while i < nnp: 
			print(time.time())
			Npcurr = Nps[np.random.choice(len(Nps), 2**i)]
			Dstars = [util.find_nearest_neighbor(patch, Npcurr) for patch in Tp]
			Dstar = (1/len(Tp))*np.sum(np.log2(Dstars))
			print(2**i, Dstar)
			xs.append(2**i)
			ys.append(Dstar)
			estimates.append([2**i, Dstar])
			i+=0.5
		fig, ax = plt.subplots()
		ax.set_xscale('log', basex=2)
		ax.plot(xs, ys)
		plt.show()

		plt.savefig(d + "_avg_nn_plot.png")
		plt.close()

	pickle.dump(estimates, open(config.gen_data_dir 
						   + n_neighbors_pow 
						   + d 
						   + config.entropy_pickle_suffix, "wb"))