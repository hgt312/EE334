import tensorflow as tf

t = tf.add(8, 9)
hello = tf.constant("Hello, Tensorflow!")
print(t)

# with tf.Session() as sess:
#     print(sess.run(t))
#
# with tf.Session() as sess:
#     print(sess.run(hello))

with tf.Session() as sess:
    matrix1 = tf.constant([[3., 3., 3.]])
    matrix2 = tf.constant([[2., 1],
                           [2., 1],
                           [2., 1]])
    product = tf.matmul(matrix1, matrix2)
    result = sess.run(product)
    print(result)

# with tf.Session() as sess:
#     x = tf.Variable(3, name='x')
#     y = tf.multiply(x, 5)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(sess.run(y))

# with tf.Session() as sess:

