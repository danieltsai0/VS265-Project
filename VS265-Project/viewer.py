import array, sys, os, math, config, pickle, util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

with open( config.gen_data_dir+'nat_data.pickle', "rb" ) as f:
	nats = pickle.load(f)
	f.close()
with open( config.gen_data_dir+'pca_data.pickle', "rb" ) as f:
	pcas = pickle.load(f)
	f.close()
with open( config.gen_data_dir+'ica_data.pickle', "rb" ) as f:
	icas = pickle.load(f)
	f.close()



for i in range(1):
	fig = plt.figure()
	a=fig.add_subplot(1,3,1)
	plt.imshow(nats[i], cmap=plt.cm.gray)
	print(nats[i])
	a.set_title("nat")
	
	a=fig.add_subplot(1,3,2)
	plt.imshow(icas[i], cmap=plt.cm.gray)
	print(icas[i])
	a.set_title("ica")
	
	a=fig.add_subplot(1,3,3)
	plt.imshow(pcas[i], cmap=plt.cm.gray)
	print(pcas[i])
	a.set_title("pca")

	plt.show()


