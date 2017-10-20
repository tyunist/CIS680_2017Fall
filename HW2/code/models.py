import tensorflow as tf
from layers import *

def quick_cnn(x, labels, c_num, batch_size, is_train, reuse, make_grad_vanish=False, resolve_grad_vanish=False):
  with tf.variable_scope('C', reuse=reuse) as vs:

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 32 
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Uncomment to see vinishing gradients
#    for l in range(8):
#      with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
#        x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)

    # conv3
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 5, 1, is_train, reuse)
      x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    
    # if is_train:
    #  x = tf.nn.dropout(x, keep_prob=0.5)
    # fc4
    with tf.variable_scope('conv4', reuse=reuse):
      x = tf.reshape(x, [batch_size, -1])
      x = fc_factory(x, hidden_num, is_train, reuse)
    feat = x

    # dropout
    if is_train:
     x = tf.nn.dropout(x, keep_prob=0.5)

    # local5
    with tf.variable_scope('fc5', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      probs = tf.nn.softmax(logits=x, name='softmax')
      confidence = tf.reduce_max(tf.one_hot(labels, c_num)*probs, reduction_indices=[1])
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))
  print 'shape of confidence', confidence.get_shape() 
  print  'shape of probs:', probs.get_shape().as_list() 
  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, confidence 


def customized_cnn(x, labels, c_num, batch_size, is_train, reuse, make_grad_vanish=False, resolve_grad_vanish=False):
  use_bn = not make_grad_vanish
  print('...Use Batchnome:', use_bn)  
  with tf.variable_scope('C', reuse=reuse) as vs:

    # conv1
    with tf.variable_scope('conv1', reuse=reuse):
      hidden_num = 128  
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse, use_bn=use_bn)
      conv1 = x 
      res_conv = x 
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # Uncomment to see vinishing gradients
    if make_grad_vanish :
      for l in range(20):
        with tf.variable_scope('rd_conv_'+str(l), reuse=reuse):
          x = conv_factory(res_conv, hidden_num, 5, 1, is_train, reuse, use_bn=use_bn)        
        if resolve_grad_vanish:
          res_conv = (x + res_conv)/2
        else:
          res_conv = x              
    # # dropout
    # if is_train and resolve_grad_vanish:
    #   res_conv = tf.nn.dropout(res_conv, keep_prob=0.5)

 
    # Create a residual connection between conv1 and conv2 
    if make_grad_vanish and resolve_grad_vanish:
      x = (res_conv + conv1)/2

   

    # # dropout
    # if is_train:
    #   res_conv = tf.nn.dropout(res_conv, keep_prob=0.5)
    # conv2
    with tf.variable_scope('conv2', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse, use_bn=use_bn)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # dropout
    if is_train:
      x = tf.nn.dropout(x, keep_prob=0.5)
    feat = x 

    # conv3
    with tf.variable_scope('conv3', reuse=reuse):
      hidden_num = 2 * hidden_num
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse, use_bn=use_bn)
      # x = tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # conv4
    with tf.variable_scope('conv4', reuse=reuse):
      x = conv_factory(x, hidden_num, 3, 1, is_train, reuse, use_bn=use_bn)
      x = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # conv5 (1x1)
    with tf.variable_scope('conv5', reuse=reuse):
      x = conv_factory(x, hidden_num, 1, 1, is_train, reuse, use_bn=use_bn)
      x = tf.nn.avg_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='VALID') # reduce features by 4 
 
    x = tf.reshape(x, [batch_size, -1])
    x_size = x.get_shape().as_list()
    hidden_num = x_size[1]
    print('x_size:', x_size)
    print('hidden_num:',hidden_num)

    # dropout
    if is_train:
      x = tf.nn.dropout(x, keep_prob=0.5)
    feat = x 

    # local4 
    with tf.variable_scope('fc4', reuse=reuse):
      W = tf.get_variable('weights', [hidden_num, c_num],
        initializer = tf.contrib.layers.variance_scaling_initializer())
      x = tf.matmul(x, W)

    # Softmax
    with tf.variable_scope('sm', reuse=reuse):
      probs = tf.nn.softmax(logits=x, name='softmax')
      confidence = tf.reduce_max(tf.one_hot(labels, c_num)*probs, reducction_indices=[1])
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=tf.one_hot(labels, c_num))
      accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(x, axis=1), labels)))

  variables = tf.contrib.framework.get_variables(vs)
  return loss, feat, accuracy, variables, confidence 
