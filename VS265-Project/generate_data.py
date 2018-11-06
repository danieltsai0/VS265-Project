import pickle, util, sys, config
import numpy as np
import matplotlib.pyplot as plt


"""
Generate 5 types of data. 
Each image is 1024 x 1024, stored as a numpy matrix. 
Each dataset is a numpy array of numpy matrices.
Serialize dataset with pickle.

Types of data:
	Gaussian, 1/f, 1/f^2, spectrum-equalized noise, natural scenes

Arguments: 
	R - generates R-by-R images
	tod - type of data to generate (nat, ica, pca)

Sample call:
	python main.py generate R nat
	python main.py generate R ica 128
	python main.py generate R pca 64

If main.py generate R is run with ica or pca, but no basis number is set,
then the basis number is set by default to 128.
"""


def generate_data(*args):

	try:
		varG, varP, data_ary = None, None, []
		# _, _, R, tod = args
		R = int(args[2])
		tod = args[3]
		if len(args) >= 5:
			basis = args[4]
		else:
			basis = ""

		if tod == "nat":
			# generate natural images
			
			data = util.crop_natural_images(config.natural_image_dir, R)

		elif tod == "ica":
			# generate reconstructions using ica
			# want to make sure to check that the natural image cropped data already exists (nat_data.pickle)
			with open( config.gen_data_dir+'nat_data.pickle', "rb" ) as f:
				images = pickle.load(f)
			if basis == "":
				basis = "128"
			data = util.ica_reconst(images, R, n=int(basis))

		elif tod == "pca":
			# generate reconstructions using pca
			# want to make sure to check that the natural image cropped data already exists (nat_data.pickle)
			with open( config.gen_data_dir+'nat_data.pickle', "rb" ) as f:
				images = pickle.load(f)
			if basis == "":
				basis = "128"
			data = util.pca_reconst(images, R, n=int(basis))

		elif tod == "gwn":
			# 3x3 patches: 127.5, 32, 513, 13
			# 8x8 patches: 127.5, 36, 513, 19 
			data = util.generate_gaussian(127.5, 32, R, 13) 

                        
		else:
			print("This type of data: {0} is undefined.".format(tod))
			sys.exit(-1)
		
		name = config.gen_data_dir + tod + basis + config.data_pickle_suffix
	
		pickle.dump(data, open(name, "wb"))

		generate_patches(R, tod, basis)

	except ValueError:
		print("incorrect command line argument formatting {0}".format(args))
		sys.exit(-1)


"""
Generate patches for a given dataset. Assumes that each image is a RxR numpy matrix.

Arguments: 
	R - size of images
	tod - type of data to load

Returns:
	tuple of patches for T and N respectively
"""
def generate_patches(R, tod, basis):

	# Compute the minimum number of neighbor images 
	Rus = R - (R % config.size_of_patches) 
	nppi = Rus**2 / config.size_of_patches**2
	mnni = int(np.ceil(np.power(2,config.max_nnpow)/nppi))

	data = util.load_dataset(tod)

	if mnni > len(data):
		print("not enough images in dataset {0} to generate patches for 2**18 neighbors".format(dataset_fn))
		sys.exit(-1)

	# Permute dataset and save patches as dictionary
	
	randomized = np.random.permutation(data)
	Ni, Ti = randomized[:mnni], randomized[mnni:]
	patch_dict = {"Tp":get_patches(Ti, R), "Np":np.random.permutation(get_patches(Ni, R))}

	# Save patches generated
	pickle.dump(patch_dict, open(config.gen_data_dir
							+ tod
							+ basis
							+ config.patches_pickle_suffix, "wb"))


"""
Generate patches for a given imagset. Assumes that each image is a R-by-R numpy matrix.

Arguments:
	imageset - array of numpy matrices
	n - size of patches to generate (n-by-n)

Returns: 
	array of n-by-n numpy matrices
"""
def get_patches(imageset, R, n=config.size_of_patches):
	
	ret, r = [], np.delete(np.arange(0, R, n), -1, 0)

	for image in imageset:
	 	ret.extend([np.array(image[x:x+n,y:y+n]) for x in r for y in r])

	return np.array(ret)
