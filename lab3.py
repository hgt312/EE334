from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf

# prepare data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])  # 28*28*1
ys = tf.placeholder(tf.float32, [None, 10])  # 0~9

# the model of the fully-connected network
W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_initializer())
b1 = tf.Variable(tf.zeros([1, 256]))
Z1 = tf.matmul(xs, W1) + b1
A1 = tf.nn.tanh(Z1)

W2 = tf.get_variable("W2", shape=[256, 10], initializer=xavier_initializer())
b2 = tf.Variable(tf.zeros([1, 10]))
Z2 = tf.matmul(A1, W2) + b2

# prediction
predictions = tf.nn.softmax(Z2)

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

