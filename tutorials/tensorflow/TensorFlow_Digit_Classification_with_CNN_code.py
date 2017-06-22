import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST/data', one_hot=True)
(train_data, validation_data, test_data) = (mnist.train, mnist.validation, mnist.test)
with tf.name_scope('Inputs') as scope:
    x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='images')
    x_image = tf.reshape(x, [-1, 28, 28, 1], name='2d_images')
    y_correct = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='correct_output')
	
# first convolution
filter_conv1 = 8
with tf.name_scope('First_convolution') as scope:
    W_conv1 = tf.truncated_normal(shape=[6, 6, 1, filter_conv1], stddev=0.1)
    b_conv1 = tf.zeros([filter_conv1])
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 2, 2, 1], padding='SAME') + b_conv1   # (None, 14, 14, 8)
    h_conv1_relu = tf.nn.relu(h_conv1)
with tf.name_scope('First_pooling') as scope:
    h_pool_1 = tf.nn.max_pool(h_conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')  # (None, 7, 7, 8)
# second convolution
filter_conv2 = 16
with tf.name_scope('Second_convolution') as scope:
    W_conv2 = tf.truncated_normal(shape=[4, 4, filter_conv1, filter_conv2], stddev=0.1)
    b_conv2 = tf.zeros([filter_conv2])
    h_conv2 = tf.nn.conv2d(h_pool_1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2   # (None, 7, 7, 16)
    h_conv2_relu = tf.nn.relu(h_conv2)
# second convolution
with tf.name_scope('Second_pooling') as scope:
    h_pool_2 = tf.nn.max_pool(h_conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')  # (None, 7, 7, 16)
sess = tf.Session()
sess.run(tf.shape(h_pool_2), feed_dict={x : train_data.images})