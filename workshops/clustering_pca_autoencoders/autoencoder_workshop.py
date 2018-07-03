# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#=================================================================
#      Data setup
#=================================================================

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#=================================================================
#      Parameters
#=================================================================

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
n_input = 784 # MNIST data input (img shape: 28*28)

#=================================================================
#      Model setup
#=================================================================

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
#------------------------------------------------------
#    TASK 1: finish the correct initialization of tensors for the dictionary
#------------------------------------------------------
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': # TODO
    'decoder_h1': # TODO
    'decoder_h2': # TODO
}
biases = {
    'encoder_b1': # TODO
    'encoder_b2': # TODO
    'decoder_b1': # TODO
    'decoder_b2': # TODO
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

#------------------------------------------------------
#     TASK 2: Building the decoder
#------------------------------------------------------
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = #TODO
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = #TODO 
    return layer_2

# Construct model
encoder_op = encoder(X)
#------------------------------------------------------
#    TASK 3: finish up the model
#------------------------------------------------------
decoder_op = #TODO

#------------------------------------------------------
#    TASK 4: prediction and label
#------------------------------------------------------
y_pred = #TODO
y_true = #TODO

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#=================================================================
#      Training
#=================================================================
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
