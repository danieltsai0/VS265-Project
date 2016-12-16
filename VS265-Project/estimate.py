import pickle, signal, sys, util, config, sys, time
import numpy as np
import matplotlib.pyplot as plt
import random

"""
Estimates entropy and relative dimension of a given dataset. 

Arguments: 
	n_neighbors_pow: used to exponentiate the number dos
	Tnum: sets size of target set (87723 for gwn, 581405 for nat in paper)
	do_entropy: if t, plots entropy, if f plots average log nearest neighbor distance
	tod: type(s) of data being measured 

Example calls:
	python main.py estimate 17 100 t nat ica
	python main.py estimate 17 100 f nat ica pca

	Note that the order of the tod doesn't matter
"""
vlog = np.vectorize(util.log)
def estimate(*args):

	# try:
		_, _, n_neighbors_pow, Tnum, is_ent, *tod = args
		estimates, nnp, Tnum = [], int(n_neighbors_pow), int(Tnum)

		for d in tod:

			if d not in config.types_of_data:
				print("Unknown type of data requested. Skipping entropy calculation for", d)
				continue

			# Load patches 
			Tp, Np = util.load_patches(d)
			totTp = len(Tp)
			totNp = len(Np)
			Tp = Tp[np.random.choice(len(Tp), Tnum)]
			Np = Np[np.random.choice(len(Np), np.power(2, nnp))]
			
			print("T: ", Tnum, "out of", totTp)
			print("N: ", 2**nnp, "out of", totNp)

			# Generate distances for each target patch
			i = 0
			k = len(Tp[0])**2
			xs, ys = [], []
			print(Tp[0])
			while i < nnp: 
				# print(time.time())
				x = 2**i
				Npcurr = Np[np.random.choice(len(Np), x)]
				Dstars = [util.find_nearest_neighbor(patch, Npcurr) for patch in Tp]
				Dstar = (1/len(Tp))*np.sum(vlog(Dstars))
				if is_ent == "t":
					entropy = util.entropy_est(k, Dstar, x)
					y = entropy/k #need entropy per pixel
					print(x, y)
				elif is_ent == "f":
					y = Dstar
					print(x, y)
				xs.append(x)
				ys.append(y)
				estimates.append([x, y])
				i+=1

			fig, ax = plt.subplots()
			ax.set_xscale('log', basex=2)
			ax.plot(xs, ys)
			plt.xlabel("Number of samples")
			
			if is_ent == "t":
				ylbl = "Entropy (bits/pixel)"
				pltstr = "_entropy_plot.png"
				suffix = config.entropy_pickle_suffix

			elif is_ent == "f":
				ylbl = "Average Log Nearest-Neighbor Distance"
				pltstr = "_avg_nn_plot.png"
				suffix = config.avgnn_pickle_suffix

			plt.ylabel(ylbl)
			plt.savefig(d + "_" + str(nnp) + "_" + str(Tnum) + pltstr)
			pickle.dump(estimates, open(config.gen_data_dir 
					   + d 
					   + "_"
					   + str(nnp)
					   + "_"
					   + str(Tnum)
					   + suffix, "wb"))

			plt.show()
			plt.close()
			
	
				
	# except ValueError:
	# 	print("incorrect command line argument formatting", args)
	# 	sys.exit(-1)


# gwnims = util.load_dataset("gwn")
# print(len(gwnims))
# util.view(gwnims[0])

# Tp, Np = util.load_patches("gwn", 18)
# print(len(Tp), len(Np))


# def avg_nn_dist(*args):
# 	_, _, n_neighbors_pow, *tod = args
# 	estimates, nnp = [], int(n_neighbors_pow)

# 	for d in tod:

# 		if d not in config.types_of_data:
# 			print("Unknown type of data requested. Skipping entropy calculation for", d)
# 			continue

# 		# Load patches 
# 		Tp, Nps = util.load_patches(d, nnp)
# 		if d == "gwn":
# 			Tp = Tp[np.random.choice(len(Tp), 87723)]
# 		elif d == "nat":
# 			Tp = Tp[np.random.choice(len(Tp), 581405)]
# 		print("T:", len(Tp))
# 		print("N:", len(Nps))
# 		# Generate distances for each target patch
# 		i = 0
# 		k = len(Tp[0])**2 #patches still in matrix form
# 		print(k)
# 		xs, ys = [], []
# 		while i < nnp: 
# 			print(time.time())
# 			Npcurr = Nps[np.random.choice(len(Nps), 2**i)]
# 			Dstars = [util.find_nearest_neighbor(patch, Npcurr) for patch in Tp]
# 			Dstar = (1/len(Tp))*np.sum(np.log2(Dstars))
# 			print(2**i, Dstar)
# 			xs.append(2**i)
# 			ys.append(Dstar)
# 			estimates.append([2**i, Dstar])
# 			i+=0.5
# 		fig, ax = plt.subplots()
# 		ax.set_xscale('log', basex=2)
# 		ax.plot(xs, ys)
# 		plt.show()

# 		plt.savefig(d + "_avg_nn_plot.png")
# 		plt.close()

# 	pickle.dump(estimates, open(config.gen_data_dir 
# 						   + n_neighbors_pow 
# 						   + d 
# 						   + config.entropy_pickle_suffix, "wb"))