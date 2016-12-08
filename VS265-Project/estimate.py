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
	python main.py estimate 17 100 nat ica
	python main.py estimate 17 100 nat ica pca

	Note that the order of the tod doesn't matter
"""
vlog = np.vectorize(util.log)
def estimate(*args):

	# try:
		_, _, n_neighbors_pow, *tod = args
		estimates, nnp, Tnum, is_ent = [], int(n_neighbors_pow)

		for d in tod:

			if d not in config.types_of_data:
				print("Unknown type of data requested. Skipping entropy calculation for", d)
				continue

			# Load patches 
			Tp, Nps = util.load_patches(d, nnp)

			Tp = Tp[np.random.choice(len(Tp), Tnum)]
		
			print("T:", len(Tp))
			print("N:", len(Nps))
			# Generate distances for each target patch
			i = 0
			k = len(Tp[0])**2
			xs, ys = [], []
			print(Tp[0])
			while i < nnp: 
				# print(time.time())
				x = 2**i
				Npcurr = Nps[np.random.choice(len(Nps), x)]
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
			plt.show()

			if is_ent == "t":
				plt.savefig(d + "_entropy_plot.png")
				pickle.dump(estimates, open(config.gen_data_dir 
						   + n_neighbors_pow
						   + d
						   + config.entropy_pickle_suffix, "wb"))
			elif is_ent == "f":
				plt.savefig(d + "_avg_nn_plot.png")
				pickle.dump(estimates, open(config.gen_data_dir 
						   + n_neighbors_pow 
						   + d 
						   + config.entropy_pickle_suffix, "wb"))
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