# from https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
from argparse import ArgumentParser
from numpy.random import randint
from numpy import zeros, ones
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot

import tensorflow as tf
import os
import gc
import numpy
import imageio

# Parse command-line arguments
parser = ArgumentParser()


parser.add_argument('--test', '-e', help='Path to test images',
					default='C:/Users/Veerle/Desktop/Seg/Seg2/test/')
					#default='C:/Users/sofie/Downloads/TSKinFace/BLUR_BASE/test/')
parser.add_argument('--path-model', '-m', help='Path to model',
					default='C:/Users/Veerle/Desktop/Seg/Seg2/models/model_32000.h5')
					#default='C:/Users/sofie/Downloads/TSKinFace/BLUR_BASE/test/')
parser.add_argument('--output', '-o', help='Path to output folder',
					default='C:/Users/Veerle/Desktop/Seg/Seg2/')
					#default='C:/Users/sofie/Downloads/TSKinFace/')
args = parser.parse_args()


# load and prepare training images
def load_real_samples(filename):
	X1_path, X2_path = os.listdir(filename + '/seg'), os.listdir(filename + '/orig')
	X1_path.sort()
	X2_path.sort()
	X1 = numpy.asarray([numpy.asarray(imageio.imread(filename + '/seg/' + j), dtype=numpy.uint8) for j in X1_path])
	X2 = numpy.asarray([numpy.asarray(imageio.imread(filename + '/orig/' + j), dtype=numpy.uint8) for j in X2_path])
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate samples and save as a plot
def summarize_performance(g_model, dataset, path_test_plots):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, len(dataset[0]), 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	
	for i in range(len(dataset[0])):
        # plot real source images
		pyplot.subplot(3, 1, 1)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
        # plot generated target image
		pyplot.subplot(3, 1, 2)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
        # plot real target image
		pyplot.subplot(3, 1, 3)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
		# save plot to file
		filename1 = path_test_plots + '/plot_%03d.png' % (i+1)
		pyplot.savefig(filename1)
		pyplot.close()


# load image data
path_test_plots = args.output + '/testplots/'
if not os.path.exists(path_test_plots):
	os.makedirs(path_test_plots)
dataset_test = load_real_samples(args.test)
print('Loaded', dataset_test[0].shape, dataset_test[1].shape)

# define the models
g_model = tf.keras.models.load_model(args.path_model, compile=False)
# evaluate model
summarize_performance(g_model, dataset_test, path_test_plots)
del(g_model)
gc.collect()
tf.keras.backend.clear_session()
