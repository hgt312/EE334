import tensorflow as tf

matrix1 = tf.constant([[3., 3., 3.],
                       [4., 4., 4.]])
matrix2 = tf.constant([[6., 6., 6.],
                       [2., 2., 2.]])

# Add
with tf.Session() as sess:
    print(sess.run(matrix2 + matrix1))

# Subtraction
with tf.Session() as sess:
    print(sess.run(tf.subtract(matrix1, matrix2)))

# Plus
with tf.Session() as sess:
    print(sess.run(3. * matrix1))

# Division
with tf.Session() as sess:
    print(sess.run(matrix1 / 4.))
