# from https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
from argparse import ArgumentParser
from numpy.random import randint
from numpy import zeros, ones
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
import os
import gc
import numpy
import imageio

# Parse command-line arguments
parser = ArgumentParser()
parser.add_argument('--train', '-i', help='Path to training images',
					default='.../train')
parser.add_argument('--test', '-e', help='Path to test images',
					default='.../test/')
parser.add_argument('--val', '-v', help='Path to validation images',
					default='.../val/')
parser.add_argument('--output', '-o', help='Path to output folder',
					default='.../output/')
parser.add_argument('--testbatch', '-t',
					help='Number of images you want to test, has to be smaller than or equal to validation dataset size',
					type=int, default=1)
parser.add_argument('--epochs', '-n', help='Number of epochs you want to train for',
					type=int, default=5)
parser.add_argument('--batches', '-b', help='Number of batches for training',
					type=int, default=64)
args = parser.parse_args()

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = tf.compat.v1.keras.initializers.RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	#d = Dropout(0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	#d = Dropout(0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	#d = Dropout(0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	#d = Dropout(0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	#d = Dropout(0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = tf.compat.v1.keras.initializers.RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	# NEW dropout layer
	#g = Dropout(0.5)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = tf.compat.v1.keras.initializers.RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape=(64,64,3)):
	# weight initialization
	init = tf.compat.v1.keras.initializers.RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

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

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, path_train_plots, path_models, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = path_train_plots + '/plot_%03d.png' % (step+1)
	pyplot.savefig(filename1, dpi=200)
	pyplot.close()
	
	av_mse = 0
	for i in range(n_samples):
		av_mse += (numpy.square(X_fakeB[i] - X_realB[i])).mean(axis=None)
	av_mse = av_mse/n_samples
	print('> Batch MSE = %.6f' % (av_mse))

	if (step+1) % 1000 == 0 and (path_models is not None):
		# save the generator model
		filename2 = path_models + '/model_%03d.h5' % (step+1)
		g_model.save(filename2)
		print('>Saved: %s and %s' % (filename1, filename2))

# train pix2pix models
def train(d_model, g_model, gan_model, dataset, dataset_test, dataset_val, path_train_plots,
	path_test_plots, path_val_plots, path_models, test_batch=5, n_epochs=4, n_batch=64):

	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	print('Start training for %d epochs in %d steps' % (n_epochs, n_steps))
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % 500 == 0 or ((i+1) % 500 == 0 and (i+1) <= 2000):
			summarize_performance(i, g_model, dataset, path_train_plots, path_models)
		gc.collect()
		if (i+1) % 1000 == 0 or ((i+1) % 500 == 0 and (i+1) <= 2000):
			summarize_performance(i, g_model, dataset_val, path_val_plots, None)
	test(n_steps, g_model, dataset_test, path_test_plots, len(dataset_test[0]))
	
def test(step, g_model, dataset, path_test_plots, n_samples=5):
	n_samples = min(n_samples, len(dataset[0]))
	# select a sample of input images
	X_realA, X_realB = dataset
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	
	for i in range(n_samples):
		fig = pyplot.figure()
		ax = pyplot.axes()
		ax.set_xlim(left=0, right=image_shape[0])
		ax.set_ylim(bottom=0, top=3*image_shape[1])
		ax.set_title('i: ' + str(i))
		im = numpy.flip(numpy.concatenate((X_realA[i], X_fakeB[i], X_realB[i]), axis=0), axis=0)
		pyplot.axis('off')
		pyplot.imshow(im)
		filename1 = path_test_plots +  '/test_' + str(i) + '_%03d.png' % (step+1)
		pyplot.savefig(filename1, dpi=200)
		pyplot.close()

# load image data
path_train_plots = args.output + '/trainplots/'
if not os.path.exists(path_train_plots):
	os.makedirs(path_train_plots)
path_models = args.output + '/models/'
if not os.path.exists(path_models):
	os.makedirs(path_models)
path_val_plots = args.output + '/valplots/'
if not os.path.exists(path_val_plots):
	os.makedirs(path_val_plots)
path_test_plots = args.output + '/testplots/'
if not os.path.exists(path_test_plots):
	os.makedirs(path_test_plots)
dataset = load_real_samples(args.train)
dataset_test = load_real_samples(args.test)
dataset_val = load_real_samples(args.val)
print('Loaded', dataset[0].shape, dataset[1].shape)
print('Loaded', dataset_test[0].shape, dataset_test[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
# define the models
d_model = define_discriminator(image_shape)
#d_model.summary()
g_model = define_generator(image_shape)
#g_model.summary()
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset, dataset_test, dataset_val, path_train_plots,
		path_test_plots, path_val_plots, path_models,
		test_batch=args.testbatch, n_epochs=args.epochs, n_batch=args.batches)
	
	
del(d_model)
del(g_model)
del(gan_model)
gc.collect()
tf.keras.backend.clear_session()
