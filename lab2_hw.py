import tensorflow as tf
import numpy as np

x_data = np.loadtxt('datax.txt')
x_data = np.reshape(x_data, (-1,))
y_data = np.loadtxt('datay.txt')
y_data = np.reshape(y_data, (-1,))

W = tf.Variable(tf.random_uniform((1,), -1., 1.))
b = tf.Variable(tf.zeros((1,)))
y = W * x_data + b

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# Before starting, initialize the variables. We will 'run' this first.
init = tf.global_variables_initializer()
# Launch the graph.
sess = tf.Session()
sess.run(init)
# Fit the line.
for step in range(12001):
    sess.run(train)
    if step % 500 == 0:
        print(step, sess.run(loss), sess.run(W), sess.run(b))
