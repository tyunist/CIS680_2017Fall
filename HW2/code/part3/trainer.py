# from __future__ import print_function                                                                 

import sys, pdb 
import os, shutil
import numpy as np
from tqdm import trange
from skimage.io import imsave
from models import *
import time, cv2 
import matplotlib.pyplot as plt 

IMG_MEAN = 121.285
IMG_STD = 64.226
def un_normalize(img):
  return  img*IMG_STD + IMG_MEAN

class Trainer(object):
  
  def __init__(self, config, img_np, correct_labels_np, wrong_labels_np):
    self.config = config
 

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

    self.img_np = np.expand_dims(img_np, 0) 
    self.correct_labels_np = np.expand_dims(correct_labels_np, 0) 
    self.wrong_labels_np  = np.expand_dims(wrong_labels_np, 0) 
     

    self.cnn_model_set = {'quick_cnn':quick_cnn, 'customized_cnn':customized_cnn}
    self.cnn_model_name= config.cnn_model
    print('...Model:', self.cnn_model_name) 

    self.lr = tf.Variable(config.lr, name='lr')

    # Question 2.3: generate gradient vanishing 
    self.make_grad_vanish = config.make_grad_vanish
    # Question 2.4: resolve gradient vanishing 
    self.resolve_grad_vanish=config.resolve_grad_vanish 

    # Exponential learning rate decay
    self.epoch_num = config.max_step / config.epoch_step
    self.decay_factor = (config.min_lr / config.lr)**(1./(self.epoch_num-1.))
    self.lr_update = tf.assign(self.lr, self.lr*self.decay_factor, name='lr_update')

    self.c_num = config.c_num

    self.model_dir = config.model_dir
    self.log_dir = config.log_dir
    self.load_path = config.load_path
    print('...Building model')
    self.x_updated = tf.placeholder(tf.float32, [1, 32, 32, 3], name='updated-x-input') 
    self.x = tf.placeholder(tf.float32, [1, 32, 32, 3], name='original-x-input') 

 
    self.correct_labels = tf.placeholder(tf.int64, [1], name='train-correct_label') 
    self.wrong_labels = tf.placeholder(tf.int64, [1], name='train-wrong_label') 

    self.build_model()
    self.build_test_model()
    print('...Create saver')
    self.saver = tf.train.Saver()

    print('...Model dir:', self.model_dir) 
    try:
      shutil.rmtree(os.path.join(self.model_dir,'adv'))
    except:
      pass 
    self.summary_writer = tf.summary.FileWriter(os.path.join(self.model_dir,'adv'))
    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_model_secs=60, # No checkpoints 
                             # global_step=self.step,
                             ready_for_local_init_op=None)

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

    print('...Restoring network')

    # Restore the network  
    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir)) 
    


  def train_adv_image(self, adv_img_number, adv_img_name, adv_img_f, result_fig_name):
    print('\n.....-----------------------------------------------.....')
    # training_log_set = np.zeros([self.max_step - self.start_step,6], dtype=np.float32)
    # save_test_accuracy = 0 
    # Load test image 
    # self.img_np = np.expand_dims(img_np, 0) 
    # self.correct_labels_np = np.expand_dims(correct_labels_np, 0) 
    # self.wrong_labels_np  = np.expand_dims(wrong_labels_np, 0) 


    x = self.img_np 
    x_updated = np.copy(self.img_np)
    x_grad = np.zeros_like(x, dtype=np.float32) 
    x_grad_updated = np.zeros_like(x, dtype=np.float32) 
    correct_labels = self.correct_labels_np
    wrong_labels = self.wrong_labels_np 
    update_rate = 1 
    start_acc = []
    try:
      for step in trange(self.start_step, self.max_step):
        if step > 200:
          break 
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir)) 
        # Update x 
        x_grad_updated = x_grad_updated +  x_grad # accumulative 
        x_updated = x + update_rate*x_grad_updated 
        # Clip x 
        x_updated = np.clip(x_updated, 0.0, 255.0) 
        x_grad_updated = (x_updated - x)/update_rate 
        fetch_dict = {
          # 'c_optim': self.c_optim,
          # 'wd_optim': self.wd_optim,
          'c_loss': self.c_loss,
          'accuracy': self.accuracy,
          'lr': self.lr,
          'confidence': self.confidence, 
          'x_grad': self.x_grad,
           }
        # Input to the grap session 
   
        feed_dict = {self.x_updated:x_updated, self.correct_labels:correct_labels, self.wrong_labels:wrong_labels}
        self.saver.save(self.sess, self.model_dir + '/model', global_step=step)
        if step % self.log_step == self.log_step - 1:
           
          fetch_dict.update({
            # 'c_optim': self.c_optim,
            # 'wd_optim': self.wd_optim,
            'c_loss': self.c_loss,
            'accuracy': self.accuracy, 
            'lr': self.lr,
            'confidence': self.confidence, 
            'x_grad': self.x_grad,
            'summary': self.summary_op })

        result = self.sess.run(fetch_dict, feed_dict)
        lr = result['lr']
        c_loss = result['c_loss']
        accuracy = result['accuracy']
    
     
        x_grad = result['x_grad']
        confidence = result['confidence']

        # print('shape of x_grad:',x_grad.shape)
        # print('shape of x_grad_updated:',x_grad_updated.shape)
      
        # training_log_set[step] = [lr, c_loss, accuracy,0, 0, save_test_accuracy]
        

        if step % self.log_step == self.log_step - 1:
          self.summary_writer.add_summary(result['summary'], step)
          self.summary_writer.flush()


          print("\n[{}/{}:{:.6f}] Loss_C: {:.6f} Accuracy: {:.4f}   " . \
                format(step, self.max_step, lr, c_loss, accuracy ))
          print("\n confidence:",confidence)       
          print("\n x_grad_total:", x_updated[0,0:10,0] - x[0,0:10,0]) 
          print('x:', np.sum(x))
          print('>>> Correct %d  Wrong %d'%( self.correct_labels_np , self.wrong_labels_np))
          print('>>> conf of correct %.4f conf of wrong %.4f total prob %.1f'%(confidence[0, self.correct_labels_np[0]], confidence[0, self.wrong_labels_np[0]], np.sum(confidence)))
          print('update_rate:', update_rate)
          sys.stdout.flush()
        source_acc = confidence[0, self.correct_labels_np[0]]
        target_acc = confidence[0, self.wrong_labels_np[0]]

        if step == 0:
          start_acc = [source_acc, target_acc] 
   
          
        if step % self.epoch_step == self.epoch_step - 1:
          self.sess.run([self.lr_update])
          # update_rate *= self.decay_factor
    except KeyboardInterrupt:
      print("Control C pressed. Saving model before exit. ")
      pass 
    # Show the example
    fig = plt.figure(figsize=(30, 10))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(self.img_np.reshape([32,32,3]).astype(np.uint8), cv2.COLOR_BGR2RGB )) 
    plt.axis('off')
    plt.title('Original. source: (%f) %d, target: (%f) %d' % (start_acc[0],int(self.correct_labels_np[0]),\
                                                              start_acc[1], int(self.wrong_labels_np[0]) ) )

    plt.subplot(132)
    plt.imshow(x_grad_updated.reshape([32,32,3]))
    plt.title('Delta (%.2f)' % np.sum(x_grad_updated))
    plt.axis('off')

    plt.subplot(133)
    x_adversarial = cv2.cvtColor(x_updated.reshape([32,32,3]).astype(np.uint8), cv2.COLOR_BGR2RGB)
    plt.imshow(x_adversarial)
    # plt.imshow(x_updated.reshape([32,32,3]))
    plt.axis('off')
    plt.title('Adversarial source: (%d), target: (%d)' % (int(self.correct_labels_np[0]), int(self.wrong_labels_np[0])))

    
  
    plt.savefig(result_fig_name)
    # plt.show()
    adv_img_f.write("{0} {1} {2}\n".format(adv_img_name, int(self.correct_labels_np[0]), int(self.wrong_labels_np[0])))
    imsave(adv_img_name, x_adversarial)
 
    # return training_log_set
    return None 

  def build_model(self, first_time=False):
    print('Wrong label:', self.wrong_labels_np[0])

    x = self.x_updated


    # Run forward part, no gradient, use wrong label. 
    self.c_loss, feat, self.accuracy, self.c_var, self.confidence = self.cnn_model_set[self.cnn_model_name](
      x, self.wrong_labels, self.c_num, self.batch_size, is_train=False, reuse=False,\
      make_grad_vanish=self.make_grad_vanish, resolve_grad_vanish=self.resolve_grad_vanish)
    self.c_loss = tf.reduce_mean(self.c_loss, 0)

    # Gather gradients of conv1 & conv4 weights for logging
    with tf.variable_scope("C/conv1", reuse=True):
      conv1_weights = tf.get_variable("weights")
    self.conv1_grad = tf.reduce_max(tf.abs(tf.gradients(self.c_loss, conv1_weights)))#, self.c_loss)))

    with tf.variable_scope("C/conv4", reuse=True):
      conv4_weights = tf.get_variable("weights")
    self.conv4_grad = tf.reduce_max(tf.abs(tf.gradients(self.c_loss, conv4_weights)))#, self.c_loss)))

    # Gradient w.r.t x 
 
    self.x_grad = tf.reshape(tf.gradients(self.confidence[0, self.wrong_labels_np[0]], x)[0],  shape= [self.batch_size, 32, 32, 3]) #, self.c_loss)

    self.x_grad = tf.sign(self.x_grad) 


    self.summary_op = tf.summary.merge([
      tf.summary.scalar("c_loss", self.confidence[0, self.wrong_labels_np[0]]),
      tf.summary.scalar("accuracy", self.confidence[0, self.wrong_labels_np[0]]),
      tf.summary.scalar("lr", self.lr),
      # tf.summary.image("inputs",  self.x),
      tf.summary.image("inputs updated",  self.x_updated),
      # tf.summary.image("x_grad", self.x_updated - self.x),

      # tf.summary.histogram("feature", feat)
    ])

  def test_valid_sample(self, img_np, correct_labels_np):
    ''' This test whether the image is classified correctly by the pretrained network'''
    # Load test image 
    self.img_np = np.expand_dims(img_np, 0) 
    self.correct_labels_np = np.expand_dims(correct_labels_np, 0) 
  
    x = self.img_np

    correct_labels = self.correct_labels_np
    # Input to the grap session 
    feed_dict = {self.x:x, self.correct_labels:correct_labels}

    test_confidence = 0
    for iter in trange(self.test_iter):
      fetch_dict = {"test_confidence":self.test_confidence}
      result = self.sess.run(fetch_dict, feed_dict)
      test_confidence += result['test_confidence']
    test_confidence /= self.test_iter

    # print("Confidence:", test_confidence) 
    return test_confidence 

  def build_test_model(self):
    x = self.x 

    loss, _, self.test_accuracy, var, self.test_confidence = self.cnn_model_set[self.cnn_model_name](
      x, self.correct_labels, self.c_num, self.batch_size_test, is_train=False, reuse=True, \
      make_grad_vanish=self.make_grad_vanish, resolve_grad_vanish=self.resolve_grad_vanish)
 
