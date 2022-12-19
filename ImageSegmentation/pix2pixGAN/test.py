# from https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
from numpy.random import randint
from numpy import zeros, ones
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot

import tensorflow as tf
import os
import gc
import numpy
import imageio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from skimage.transform import resize

# Parse command-line arguments

def test_pix2pix(test, path_model, output):
    # load and prepare training images
    def load_real_samples(dataset):
        
        X1 = numpy.asarray(dataset[0])
        X2 = numpy.asarray(dataset[1])
        X1 = resize(X1, (X1.shape[0], 256, 256, 3), anti_aliasing=True)
        X2 = resize(X2, (X2.shape[0], 256, 256, 3), anti_aliasing=True)
        #pyplot.imshow(X1[0]/255)
        #pyplot.show()
        #pyplot.imshow(X2[0]/255)
        #pyplot.show()
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
        # retrieve selected images
        X1, X2 = trainA, trainB
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
        
        return X_fakeB


    # load image data
    path_test_plots = output + '/testplots/'
    if not os.path.exists(path_test_plots):
        os.makedirs(path_test_plots)
    dataset_test = load_real_samples(test)
    print('Loaded', dataset_test[0].shape, dataset_test[1].shape)

    # define the models
    g_model = tf.keras.models.load_model(path_model, compile=False)
    # evaluate model
    data_result = summarize_performance(g_model, dataset_test, path_test_plots)
    del(g_model)
    gc.collect()
    tf.keras.backend.clear_session()
    return data_result
