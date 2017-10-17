# from __future__ import print_function                                                                 

import sys
import os
import numpy as np
from tqdm import trange

from models import *

def norm_img(img):
  return img / 127.5 - 1.

def denorm_img(img):
  return (img + 1.) * 127.5

# def flip_flip(img):


class Trainer(object):
  def __init__(self, config, data_loader, label_loader, test_data_loader, test_label_loader):

    self.config = config
    self.data_loader = data_loader
    self.label_loader = label_loader
    self.test_data_loader = test_data_loader
    self.test_label_loader = test_label_loader

    self.optimizer = config.optimizer
    self.batch_size = config.batch_size
    self.batch_size_test = config.batch_size_test

    self.step = tf.Variable(0, name='step', trainable=False)
    self.start_step = 0
    self.log_step = config.log_step
    self.epoch_step = config.epoch_step
    self.max_step = config.max_step
    self.save_step = config.save_step
    self.test_iter = config.test_iter
    self.wd_ratio = config.wd_ratio

    self.cnn_model_set = {'quick_cnn':quick_cnn, 'customized_cnn':customized_cnn}
    self.cnn_model_name= config.cnn_model
    print('...Model:', self.cnn_model_name) 

    self.lr = tf.Variable(config.lr, name='lr')

    # Question 2.3: generate gradient vanishing 
    self.make_grad_vanish = config.make_grad_vanish

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    decay_factor = (config.min_lr / config.lr)**(1./(self.epoch_num-1.))
    self.lr_update = tf.assign(self.lr, self.lr*decay_factor, name='lr_update')

    self.c_num = config.c_num

    self.model_dir = config.model_dir
    self.load_path = config.load_path
    print('...Building model')
    self.build_model()
    self.build_test_model()
    print('...Create saver')
    self.saver = tf.train.Saver()

    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_model_secs=60,
                             global_step=self.step,
                             ready_for_local_init_op=None)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    print('...Model_dir:', self.model_dir)
    training_log_set = np.zeros([self.max_step - self.start_step,6], dtype=np.float32)
    for step in trange(self.start_step, self.max_step):
      fetch_dict = {
        'c_optim': self.c_optim,
        'wd_optim': self.wd_optim,
        'c_loss': self.c_loss,
        'accuracy': self.accuracy,
        'lr': self.lr,
        'conv1_grad': self.conv1_grad,
        'conv4_grad': self.conv4_grad }

      if step % self.log_step == self.log_step - 1:
        fetch_dict.update({
          'c_optim': self.c_optim,
          'wd_optim': self.wd_optim,
          'c_loss': self.c_loss,
          'accuracy': self.accuracy, 
          'lr': self.lr,
          'conv1_grad': self.conv1_grad,
          'conv4_grad': self.conv4_grad, 
          'summary': self.summary_op })

      result = self.sess.run(fetch_dict)
      lr = result['lr']
      c_loss = result['c_loss']
      accuracy = result['accuracy']
      conv1_grad = result['conv1_grad']
      conv4_grad = result['conv4_grad']
      training_log_set[step] = [lr, c_loss, accuracy, conv1_grad, conv4_grad, 0]

      if step % self.log_step == self.log_step - 1:
        self.summary_writer.add_summary(result['summary'], step)
        self.summary_writer.flush()

        lr = result['lr']
        c_loss = result['c_loss']
        accuracy = result['accuracy']

        print("\n[{}/{}:{:.6f}] Loss_C: {:.6f} Accuracy: {:.4f}" . \
              format(step, self.max_step, lr, c_loss, accuracy))
        sys.stdout.flush()

      if step % self.save_step == self.save_step - 1:
        self.saver.save(self.sess, self.model_dir + '/model',global_step=step)
        
        test_accuracy = 0
        for iter in xrange(self.test_iter):
          fetch_dict = { "test_accuracy":self.test_accuracy }
          result = self.sess.run(fetch_dict)
          test_accuracy += result['test_accuracy']
        test_accuracy /= self.test_iter

        print("\n[{}/{}:{:.6f}] Test Accuracy: {:.4f}" . \
              format(step, self.max_step, lr, test_accuracy))
        sys.stdout.flush()


      if step % self.epoch_step == self.epoch_step - 1:
        self.sess.run([self.lr_update])
    self.saver.save(self.sess, self.model_dir + '/model', global_step=step)
    return training_log_set 

  def build_model(self):
    self.x = self.data_loader
    self.labels = self.label_loader
    # if self.is_normalize:
    #   print '....Mode: normalize'
    #   x = norm_img(self.x)
    # else:
    x = self.x 
      # print '....Mode: Do not normalize'


    self.c_loss, feat, self.accuracy, self.c_var = self.cnn_model_set[self.cnn_model_name](
      x, self.labels, self.c_num, self.batch_size, is_train=True, reuse=False,\
      make_grad_vanish=self.make_grad_vanish)
    self.c_loss = tf.reduce_mean(self.c_loss, 0)

    # Gather gradients of conv1 & conv4 weights for logging
    with tf.variable_scope("C/conv1", reuse=True):
      conv1_weights = tf.get_variable("weights")
    self.conv1_grad = tf.reduce_max(tf.abs(tf.gradients(self.c_loss, conv1_weights)))#, self.c_loss)))

    with tf.variable_scope("C/conv4", reuse=True):
      conv4_weights = tf.get_variable("weights")
    self.conv4_grad = tf.reduce_max(tf.abs(tf.gradients(self.c_loss, conv4_weights)))#, self.c_loss)))

    # Gradient w.r.t x 
    x_grad = tf.gradients(self.c_loss, x) #, self.c_loss)
    x_grad = tf.reduce_sum(tf.abs(x_grad[0]), 3, True)
    x_grad = (x_grad - tf.reduce_min(x_grad)) / (tf.reduce_max(x_grad) - tf.reduce_mean(x_grad))
    self.x_grad = tf.multiply(self.x , x_grad)

    wd_optimizer = tf.train.GradientDescentOptimizer(self.lr)
    if self.optimizer == 'sgd':
      c_optimizer = tf.train.MomentumOptimizer(self.lr, 0.9)
    elif self.optimizer == 'adam':
      c_optimizer = tf.train.AdamOptimizer(self.lr)
    else:
      raise Exception("[!] Caution! Don't use {} opimizer.".format(self.optimizer))

    for var in tf.trainable_variables():
      weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd_ratio)
      tf.add_to_collection('losses', weight_decay)
    wd_loss = tf.add_n(tf.get_collection('losses'))

    self.c_optim = c_optimizer.minimize(self.c_loss, var_list=self.c_var)
    self.wd_optim = wd_optimizer.minimize(wd_loss)

    self.summary_op = tf.summary.merge([
      tf.summary.scalar("c_loss", self.c_loss),
      tf.summary.scalar("accuracy", self.accuracy),
      tf.summary.scalar("lr", self.lr),
      tf.summary.scalar("conv1_grad", self.conv1_grad),
      tf.summary.scalar("conv4_grad", self.conv4_grad),

      tf.summary.image("inputs", self.x),
      tf.summary.image("x_grad", self.x_grad),

      tf.summary.histogram("feature", feat)
    ])

  def test(self):
    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir)) 
    test_accuracy = 0
    for iter in trange(self.test_iter):
      fetch_dict = {"test_accuracy":self.test_accuracy}
      result = self.sess.run(fetch_dict)
      test_accuracy += result['test_accuracy']
    test_accuracy /= self.test_iter

    print("Accuracy: {:.4f}" . format(test_accuracy))

  def build_test_model(self):
    self.test_x = self.test_data_loader
    self.test_labels = self.test_label_loader
    # if self.is_normalize:
    #   test_x = norm_img(self.test_x)
    # else:
    test_x = self.test_x 

    loss, self.test_feat, self.test_accuracy, var = self.cnn_model_set[self.cnn_model_name](
      test_x, self.test_labels, self.c_num, self.batch_size_test, is_train=False, reuse=True, \
      make_grad_vanish=self.make_grad_vanish)
