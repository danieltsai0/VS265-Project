import numpy as np
import sys, os, array
import matplotlib.pyplot as plt

ndir = "..\\Data\\vanhateren_iml\\"

filename = os.listdir(ndir)[0]
fin = open( ndir+filename, 'rb' )
s = fin.read()
fin.close()
arr = array.array('H', s)
arr.byteswap()
#img = np.array(arr, dtype='uint8').reshape(1024,1536)
img = np.array(arr, dtype='f').reshape(1024,1536)
img8 = np.int8(img)
print(np.max(img8))
# plt.figure()
# plt.imshow(img8, cmap='gray')
# plt.show()

# with open("filenums.txt", "r") as f:
# 	print(f.readline())