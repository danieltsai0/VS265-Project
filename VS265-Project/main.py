import pickle, signal, sys
import numpy as np
import matplotlib.pyplot as plt

from util import load_dataset
from train import train_model

# objects to be pickled: 
# 	learned model
# 	iterator over dataset


"""
Main method that either loads or creates a new model that is to be 
trained. 

Arguments: 
	model_fn: name of file containing serialized (pickled) model.
	dataset_fn: name of file containing dataset. 
"""
def main(model_fn=None, dataset_fn = ""):

	if model_fn == None:
		# do stuff to create new model
		model = create_new_model("params")
	else:
		# do stuff to load old model 
		with open(model_fn, "rb") as f:
			model = pickle.load(model_fn)
	# train
	data = load_dataset(dataset_fn)
	train(model, "other params")




if __name__ == "__main__":

	if len(sys.argv) > 1:
		main(sys.argv[1])
	else:
		main()