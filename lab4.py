from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf

# prepare data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])  # 28*28*1
ys = tf.placeholder(tf.float32, [None, 10])  # 0~9

input = tf.reshape(xs, [-1, 28, 28, 1])

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input,
    filters=32,
    kernel_size=[5, 5],
    padding="valid",
    activation=tf.nn.relu)

# Convolutional Layer #2
conv2 = tf.layers.conv2d(
    inputs=conv1,
    filters=32,
    kernel_size=[5, 5],
    padding="valid",
    activation=tf.nn.relu)

# Convolutional Layer #3
conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=64,
    kernel_size=[5, 5],
    padding="valid",
    activation=tf.nn.relu)

# Convolutional Layer #4
conv4 = tf.layers.conv2d(
    inputs=conv3,
    filters=64,
    kernel_size=[5, 5],
    padding="valid",
    activation=tf.nn.relu)

# Convolutional Layer #5
conv5 = tf.layers.conv2d(
    inputs=conv4,
    filters=128,
    kernel_size=[5, 5],
    padding="valid",
    activation=tf.nn.relu)

# Convolutional Layer #6
conv6 = tf.layers.conv2d(
    inputs=conv5,
    filters=256,
    kernel_size=[5, 5],
    padding="valid",
    activation=tf.nn.relu)

pool = tf.layers.max_pooling2d(inputs=conv6, pool_size=[4, 4], strides=4)

flat = tf.reshape(pool, [-1, 256])

dense = tf.layers.dense(inputs=flat, units=10)

# prediction
predictions = tf.nn.softmax(dense)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predictions), reduction_indices=[1]))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=ys)

train_step = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)

# compute the accuracy
correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(ys, 1))

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(256)

        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

        if (i + 1) % 100 == 0:
            print("steps : %d " % (i + 1), "accuracy: ", sess.run(accuracy, feed_dict={
                xs: mnist.test.images,
                ys: mnist.test.labels
            }))
