from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import os, pdb
import scipy.misc
import tensorflow as tf
from sklearn.cluster import KMeans

from utils import *
from dataLoader import *
import matplotlib.pyplot as plt 
# Set one GPU 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)

################
# Define flags #
################

home_dir = '/home/tynguyen/cis680/HW4'

flags = tf.app.flags
flags.DEFINE_string("logdir", home_dir+'/logs/2.1/', "Directory to save logs")
flags.DEFINE_string("sampledir", home_dir+'/out/2.1/', "Directory to save samples")
flags.DEFINE_boolean("classifier", False, "Use the discriminator for classification")
flags.DEFINE_boolean("kmeans", False, "Run kmeans of intermediate features")
flags.DEFINE_integer("batch_size", 36, "The size of batch images [32]")
flags.DEFINE_integer("train_steps_model_D", 2, "The training iterations of model D [3]")
flags.DEFINE_integer("interval_plot", 100, "The step interval to plot generative images [10]")
flags.DEFINE_boolean("debug", False, "True if debug mode")
flags.DEFINE_boolean("resume", False, "True if resuming the training")
flags.DEFINE_boolean("fix_sample", True, "True if want to use one fixed sample over time")
flags.DEFINE_boolean("max_steps", int(2), "Max number of training steps")
FLAGS = flags.FLAGS

# Constants 
z_dim = 100
# A fixed noise used to generate samples 
Z2 = np.random.uniform(-1.0, 1.0, size=[FLAGS.batch_size, z_dim]).astype(np.float32)

checkpoint_file = os.path.join(FLAGS.logdir, 'checkpoint')

if not FLAGS.resume:
  try:
    print('===> Retrain, removing %s'%FLAGS.sampledir)
    shutil.rmtree(FLAGS.sampledir)
  except:
    print('===> Resuming...%s'%FLAGS.sampledir)
    pass 
  try:
    print('===> Retrain, removing %s'%FLAGS.logdir)
    shutil.rmtree(FLAGS.logdir)
  except:
    print('===> Resuming...%s'%FLAGS.logdir)
    pass 
if not os.path.exists(FLAGS.sampledir):
  os.makedirs(FLAGS.sampledir)
###############
# DCGAN Model #
###############

'''
  Generative Model G
  - Input z: random noise vector
  - Output net: generated map with same size as data
'''
def generator(z):
  # the output matrix size after reshape
  init_height, init_width = 4, 4
  channel_num = (1024, 512, 256, 128, 1)

  with tf.variable_scope("Model_G") as scope:
    # Reshape input z as 10 x 10 image
    noise_img =tf.reshape(z, [-1, 10, 10]) 
    # fc converts noise vector z into required size 
    net = linear(z, init_height * init_width * channel_num[0], 'g_fc')
    # reshape feature vector into matrix with size [bs, height, width, channel]
    net = tf.reshape(net, [-1, init_height, init_width, channel_num[0]])
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn0'))

    # 1st deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 8, 8, channel_num[1]], name='g_deconv1')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn1'))    

    # 2nd deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 16, 16, channel_num[2]], name='g_deconv2')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn2'))    

    # 3rd deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 32, 32, channel_num[3]], name='g_deconv3')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn3'))    

    # 4th deconvolution (upsample by 2)
    net = deconv2d(net, output_shape=[FLAGS.batch_size, 64, 64, channel_num[-1]], name='g_deconv4')
    net = tf.nn.relu(batch_norm(net, 0.9, 1e-5, 'g_bn4'))    

    # nonlinearize
    net = tf.nn.tanh(net)

    # tf.histogram_summary('Model_G/out', net)
    tf.summary.histogram('Model_G/out', net)
    tf.summary.histogram('Model_G/in', noise_img)
    # tf.image_summary("Model_G", net, max_images=8)

  return net

'''
  Discriminator model D
  - Input net: image data from dataset or the G model
  - Output prob: the scalar to represent the prob that net belongs to the real data
'''
def discriminator(net, reuse=False):
  with tf.variable_scope("Model_D") as scope:
    if reuse:
      scope.reuse_variables()

    # simulate the inverse operation of the generative model G
    channel_num = (128, 256, 512, 1024)

    # standard convolutional layer
    # 1st convolutional layer
    feaMap = lrelu(conv2d(net, output_dim=channel_num[0], name='d_conv0'))

    #2nd convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[1], name='d_conv1')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn1'))

    # 3rd convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[2], name='d_conv2')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn2'))

    # 4th convolutional layer
    feaMap = conv2d(feaMap, output_dim=channel_num[3], name='d_conv3')
    feaMap = lrelu(batch_norm(feaMap, 0.9, 1e-5, 'd_bn3'))

    # reshape feature map and use fc to get 1 size output as prob
    prob = linear(tf.reshape(feaMap, [FLAGS.batch_size, -1]), 1, 'd_fc')

    # apply sigmoid for prob computation
    return tf.nn.sigmoid(prob)
   


####################
# DCGAN main train #
####################
def DCGAN_main(dataset):
  # Loss storgage
  total_losses = np.zeros([FLAGS.max_steps, 3])
 
  # Models
  x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 64, 64, 1])
  d_model = discriminator(x, reuse=False)

  z = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, z_dim])  
  g_model = generator(z)
  dg_model = discriminator(g_model, reuse=True)

  # Optimizers
  t_vars = tf.trainable_variables()
  global_step = tf.Variable(0, name='global_step', trainable=False)
  d_loss = -tf.reduce_mean(tf.log(d_model) + tf.log(1. - dg_model))
  tf.summary.scalar('d_loss', d_loss)

  # optimizer for model D training
  lr_modelD = .000002
  lr_modelG = 15 * lr_modelD

  d_trainer = tf.train.AdamOptimizer(lr_modelD, beta1=.5).minimize(
      d_loss, global_step=global_step, var_list=[v for v in t_vars if 'Model_D/' in v.name])

  # optimizer for model G training
  g_loss = tf.reduce_mean(1 - tf.log(dg_model))
  # g_loss = tf.reduce_mean(1. - tf.log(dg_model_train_g))
  tf.summary.scalar('g_loss', g_loss)
  g_trainer = tf.train.AdamOptimizer(lr_modelG, beta1=.5).minimize(
      g_loss, var_list=[v for v in t_vars if 'Model_G/' in v.name])

  # Session
  sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False, device_count={'GPU':1}))
  sess.run(tf.global_variables_initializer())


  # Savers
  saver = tf.train.Saver(max_to_keep=20)
  checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
  if checkpoint and not FLAGS.debug and FLAGS.resume:
    print('Restoring from', checkpoint)
    saver.restore(sess, checkpoint)
    start_step = global_step.eval(session=sess)    
  else:
    start_step = 0     
  summary = tf.summary.merge_all()
  #summary = tf.merge_all_summaries()
  summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)


  # Training loop
  for step in range(start_step, 2 if FLAGS.debug else start_step + FLAGS.max_steps):    
    # generate z noise
    z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)

    # data x random shuffle 
    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    data_step = dataset[arr[:], :, :, :]
    x_batch = data_step[0 : FLAGS.batch_size]


    # update model D for k times
    d_loss_val = 0
    for k in range(FLAGS.train_steps_model_D):
      # Update discriminator
      _, d_loss_val_cur = sess.run([d_trainer, d_loss], feed_dict={x: x_batch, z: z_batch})
      d_loss_val += d_loss_val_cur

    d_loss_val /= FLAGS.train_steps_model_D

    # update model G for one time 
    sess.run(g_trainer, feed_dict={z: z_batch})
    _, g_loss_val = sess.run([g_trainer, g_loss], feed_dict={z: z_batch})
    
    # Dump losses
    total_losses[step-start_step, 0] = step
    total_losses[step-start_step, 1] = g_loss_val
    total_losses[step-start_step, 2] = d_loss_val

    # Log details
    print ('====> The {}th training step || Model G Loss: {:.8f} || Model D Loss: {:.8f}'.format(step, g_loss_val, d_loss_val))
    if step % 1000 == 0:
      summary_str = sess.run(summary, feed_dict={x: x_batch, z: z_batch})
      summary_writer.add_summary(summary_str, global_step.eval())

    # Early stopping
    if np.isnan(g_loss_val) or np.isnan(g_loss_val):
      print('Early stopping')
      break
    # plot generative images
    if step % FLAGS.interval_plot == 0:
      # Save samples
      if FLAGS.sampledir:
          samples = FLAGS.batch_size
          if FLAGS.fix_sample:
            z2 = Z2
          else:
            z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
          images = sess.run(g_model, feed_dict={z: z2})
          images = np.reshape(images, [samples, 64, 64]).astype(np.float32)
          images = (images + 1.) / 2.
          scipy.misc.imsave(FLAGS.sampledir + 'sample_%d.png'%step, merge(images, [int(math.sqrt(samples))] * 2))
    if step % 1000 == 0:
      # save model
      if not FLAGS.debug:
          saver.save(sess, checkpoint_file, global_step=global_step)
  # Dump losses
  
  fig_name = os.path.join(FLAGS.logdir, 'result_fig.png')
  plt.figure(figsize=(10,10))
  plt.subplot(1,2,1)
  plt.plot(total_losses[:, 1])
  plt.title('G Loss')
  plt.xlabel('Iterations') 
  plt.ylabel('Loss') 
  plt.subplot(1,2,2)
  plt.plot(total_losses[:, 2])
  plt.title('D Loss')
  plt.xlabel('Iterations') 
  plt.ylabel('Loss') 
  plt.savefig(fig_name) 
 
  result_file_name = os.path.join(FLAGS.logdir, 'result_output.txt')
  with open(result_file_name, 'wb') as f:
    np.savetxt(f, total_losses)
    print('===> Successfuly write results to %s (step, g_loss, d_loss)'%result_file_name)
  print('===> End!===============================') 
   
  return


########
# Main #
########
def main():
  # load data
  train_dict_cufs = dataProcess()
  test_dict_cufs = dataProcess(True)

  train_imgs_cufs, train_ind_cufs = train_dict_cufs['img'], train_dict_cufs['order']
  test_imgs_cufs, test_ind_cufs = test_dict_cufs['img'], test_dict_cufs['order']

  DCGAN_main(train_imgs_cufs)

  # if not tf.gfile.Exists(FLAGS.logdir):
  #     tf.gfile.MakeDirs(FLAGS.logdir)
  # if FLAGS.sampledir and not tf.gfile.Exists(FLAGS.sampledir):
  #     tf.gfile.MakeDirs(FLAGS.sampledir)
  # if FLAGS.sampledir:
  #     sample()
  #     return
  # dataset = input_data.read_data_sets('MNIST_data', one_hot=True)
  # if FLAGS.classifier:
  #     gan_class(dataset)
  # elif FLAGS.kmeans:
  #     kmeans(dataset)
  # else:
  #     mnist_gan(dataset)


if __name__ == '__main__':
  # tf.app.run()
  main() 
