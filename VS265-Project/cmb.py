import pickle, signal, sys, util, config, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

"""
This file specially handles entropy computation for 1 specific image of the
Cosmic Microwave Background Radiation.  The European Space Agency launched 
the Planck satellite to complete a mission lasting from 2009-2013 to accurately 
measure anistropies in the CMB.

Source: http://planck.cf.ac.uk/all-sky-images

Image: PlanckFig_BW_map_no-out_12000px_CMB_cart.png

"""


def generate_cmb_patches():
	"""Preprocess CMB like other images, real values 0-255", averaged all 3 BW channels (shouldn't change anything)"""
	cmb = mpimg.imread("PlanckFig_BW_map_no-out_12000px_CMB_cart.png")
	cmb = cmb[:,:,0]
	# cmb = (cmb[:,:,0] + cmb[:,:,1] + cmb[:,:,2])/(3.)
	cmb = cmb*(255./np.max(cmb))

	"""Crop to center 2000x4000"""
	print(cmb.shape)
	cmb = cmb[3000:4998,4000:7999]
	print(cmb.shape)
	# print(cmb.shape)
	# plt.imshow(cmb)
	# plt.show
	"""Generate ((2000x4000) / (nxn)) number of nxn patches"""
	ret, n = [], 3 
	r = np.arange(0, cmb.shape[0], n)
	s = np.arange(0, cmb.shape[1], n)
	for x in r:
		for y in s:
			patch = cmb[x:x+n,y:y+n]
			ret.append(patch)
			if x%100==0 and y%100==0:
				print(patch)
	with open(config.gen_data_dir + "cmb" + config.patches_pickle_suffix, "wb") as f:
		pickle.dump(ret, f)
	return

"""Randomly chooses Np and Tp, since there's only one image to work with here."""
def cmb_entropy(nnp,Tnum):
	is_ent = "t"
	d = "cmb"
	with open(config.gen_data_dir + "cmb" + config.patches_pickle_suffix, "rb") as f:
		cmb = pickle.load(f)	
	
	np.random.shuffle(cmb)
	cmb = np.array(cmb)
	Tp = cmb[0:Tnum]
	Np = cmb[Tnum:Tnum+(2**nnp)]

	print("T: ", Tnum, "out of", len(cmb))
	print("N: ", 2**nnp, "out of", len(cmb) - Tnum)
	print(Tp[0])

	i = 0
	k = len(Tp[0])**2
	num_neighbor_vec, entropy_vec, avg_log_nn_vec = [], [], []

	for num_neighbors in np.power(2,np.arange(nnp)):
		# 
		Npcurr = Np[np.random.choice(len(Np), num_neighbors)]
		Dstars = [util.find_nearest_neighbor(patch, Npcurr) for patch in Tp]

		avg_log_nn = (1/len(Tp))*np.sum(vlog(Dstars))
		entropy = util.entropy_est(k, avg_log_nn, num_neighbors) / k
		
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

vlog = np.vectorize(util.log)
# generate_cmb_patches()
cmb_entropy(19, 500)