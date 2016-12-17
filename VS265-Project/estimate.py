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
			print(Tp[0])

			# Generate distances for each target patch
			i = 0
			k = len(Tp[0])**2
			num_neighbor_vec, entropy_vec, avg_log_nn_vec = [], [], []

			for num_neighbors in np.power(2,np.arange(nnp)):

				Npcurr = Np[np.random.choice(len(Np), num_neighbors)]
				Dstars = [util.find_nearest_neighbor(patch, Npcurr) for patch in Tp]

				avg_log_nn = (1/len(Tp))*np.sum(vlog(Dstars))
				entropy = util.entropy_est(k, avg_log_nn, num_neighbors) / k if is_ent=="t" else Dstar
				
				num_neighbor_vec.append(num_neighbors)
				entropy_vec.append(entropy)
				avg_log_nn_vec.append(avg_log_nn)

				print(num_neighbors,entropy)

			# Set vars based on requested graph
			if is_ent == "t":
				ylbl = "Entropy (bits/pixel)"
				pltstr = "_entropy_plot.png"
				title = "Entropy vs. Number of Samples for " + str(d) + " data, N: 2^" + str(nnp) + ", T: " + str(Tnum)
				ys = entropy_vec

			elif is_ent == "f":
				ylbl = "Average Log NN Distance"
				pltstr = "_avg_nn_plot.png"
				title = ylbl + " vs. Number of Samples for " + str(d) + " data, N: 2^" + str(nnp) + ", T: " + str(Tnum)
				ys = avg_log_nn_vec


			# Plot data
			fig, ax = plt.subplots()
			ax.set_xscale('log', basex=2)
			ax.plot(num_neighbor_vec, ys)
			plt.xlabel("Number of samples")
			plt.ylabel(ylbl)
			plt.title(title)
			#save .png of this graph
			plt.savefig(d + "_" + str(nnp) + "_" + str(Tnum) + pltstr)
			plt.show()
			plt.close()

			# Pickle entropy data
			pickle.dump([entropy_vec, num_neighbor_vec], open(config.gen_data_dir 
																	  + d 
																	  + "_"
																	  + str(nnp)
																	  + "_"
																	  + str(Tnum)
																	  + config.entropy_pickle_suffix, "wb"))

			# Pickle avg nn data
			pickle.dump([avg_log_nn_vec, num_neighbor_vec], open(config.gen_data_dir 
																	  + d 
																	  + "_"
																	  + str(nnp)
																	  + "_"
																	  + str(Tnum)
																	  + config.avgnn_pickle_suffix, "wb"))

			# Pickle 
	
	
				
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