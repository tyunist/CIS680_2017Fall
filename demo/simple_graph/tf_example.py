import numpy as np
np.random.seed(0)
import tensorflow as tf

n1, n2, n3 = 8, 4, 2

# Computation graph
x = tf.placeholder(tf.float32, [n1, n2])
y = tf.placeholder(tf.float32, [n2, n3])
z = tf.placeholder(tf.float32, [n1, n3])

a = tf.matmul(x, y)
b = a + z
c = tf.reduce_mean(b)

# Calculate gradients
grad_x, grad_y, grad_z = tf.gradients(c, [x, y, z])

# Run graph
with tf.Session() as sess:
  feed_dict = {x: np.random.randn(n1, n2),
               y: np.random.randn(n2, n3),
               z: np.random.randn(n1, n3)}
  out = sess.run([c, grad_x, grad_y, grad_z], feed_dict=feed_dict)

# Print results
print("Output c = %f" % out[0])
print("Mean gradients of (x, y, z) = (%f, %f, %f)" % \
  (np.mean(out[1]), np.mean(out[2]), np.mean(out[3])))

