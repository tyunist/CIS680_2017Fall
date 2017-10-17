import tensorflow as tf
slim = tf.contrib.slim

def conv_factory(x, hidden_num, kernel_size, stride, is_train, reuse, use_bn=True):
  vs = tf.get_variable_scope()
  in_channels = x.get_shape()[3]
  W = tf.get_variable('weights', [kernel_size,kernel_size,in_channels,hidden_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, 1, 1, hidden_num],
        initializer = tf.constant_initializer(0.0))

  x = tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')
  if use_bn:
    x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True,
        fused=True, scope=vs, updates_collections=None)
  #x = tf.nn.relu(x)
  x = tf.nn.sigmoid(x)
  return x

def fc_factory(x, hidden_num, is_train, reuse, use_bn=True):

  vs = tf.get_variable_scope()
  in_channels = x.get_shape()[1]
  W = tf.get_variable('weights', [in_channels,hidden_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
  b = tf.get_variable('biases', [1, hidden_num],
        initializer = tf.constant_initializer(0.0))

  x = tf.matmul(x, W)
  if use_bn:
    x = slim.batch_norm(x, is_training=is_train, reuse=reuse, scale=True,
        fused=True, scope=vs, updates_collections=None)
  x = tf.nn.relu(x)
#  x = tf.nn.sigmoid(x)
  return x

def leaky_relu(x):
  alpha = 0.05
  pos = tf.nn.relu(x)
  neg = alpha * (x - abs(x)) * 0.5
  return pos + neg
