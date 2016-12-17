import array, sys, os, math, config, pickle, util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


tod, datas = ["ica128", "ica32", "ica16", "ica8"], []
for d in tod:
	with open(config.gen_data_dir+d+config.data_pickle_suffix, "rb" ) as f:
		datas.append(pickle.load(f))

fig=plt.figure()
for i in range(len(datas)):
	a=fig.add_subplot(1,len(datas),i+1)
	plt.imshow(datas[i][0], cmap=plt.cm.gray)
	print(datas[i][0])
	a.set_title(tod[i])

plt.show()

