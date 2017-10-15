from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import time, os
import seaborn as sns
import tensorflow as tf 
sns.set(color_codes=True)

def draw_accu_loss(accuracy_set, loss_set):
    plt.figure(figsize=(12, 5))
    fig = plt.subplot(121)
    plt.plot(range(len(accuracy_set)), accuracy_set)
    fig.set_title('Accuracy')
    fig.set_xlabel('Iterations')
    fig.set_ylabel('Accuracy (%)')

    fig = plt.subplot(122)
    plt.plot(range(len(loss_set)), loss_set)
    fig.set_title('Loss')
    fig.set_xlabel('Iterations')
    fig.set_ylabel('Loss')
    plt.show()

class FNN(object):
    @classmethod
    def init_weights(clc, shape, name):
        """ Weight initialization """
        weights = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(weights, name=name)
    @classmethod 
    def init_biases(clc, shape,  name):
        """ Bias initialization """
        biases = 0.1*tf.ones(shape, dtype=tf.float32)
        return tf.Variable(biases, name=name)
    @classmethod 
    def sigmoid_cross_entropy(clc, labels, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_loss')
        return tf.reduce_mean(loss)
    @classmethod 
    def sigmoid_activation(clc, x):
        return tf.nn.sigmoid(x) 
    @classmethod
    def relu_activation(clc, x):
        return tf.nn.relu(x)
    @classmethod 
    def sigmoid_L2(clc, labels, logits): 
        out_fc2 = tf.nn.sigmoid(logits) 
        loss = tf.square(out_fc2 - labels) 
        loss = tf.reduce_mean(loss) 
        return loss 
    
    def __init__(self, activation_fnc, loss_fnc):
        activation_function_list=  {'sigmoid_activation':self.sigmoid_activation, 'relu_activation':self.relu_activation}
        loss_function_list = {'sigmoid_cross_entropy':self.sigmoid_cross_entropy, 'sigmoid_L2':self.sigmoid_L2}
        self.activation = activation_function_list[activation_fnc] if activation_fnc else self.sigmoid_activation 
        self.loss = loss_function_list[loss_fnc] if loss_fnc else self.sigmoid_cross_entropy
        
            
    def train(self, images, labels): 
        # Initialize a network 
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                x = tf.placeholder(tf.float32, [None,4,4], name='x-input')
                y = tf.placeholder(tf.float32, [None,2], name='labels')
        # Reshape x 
        x_flat = tf.contrib.layers.flatten(x)
        with tf.name_scope('FC1'):
            w1 = self.init_weights((16,4), 'w_fc1')
            b1 = self.init_biases((4), 'b_fc1')
            with tf.name_scope('act_fc1'):
                out_fc1 = self.activation(tf.matmul(x_flat , w1) + b1)

        with tf.name_scope('FC2'):
            w2 = self.init_weights((4,2), 'w_fc2')
            b2 = self.init_biases((2), 'b_fc2')
            logits = tf.matmul(out_fc1, w2) + b2
            predict = tf.argmax(logits, axis=1)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(logits,1))
            loss = self.loss(labels=y, logits=logits)
         

        # Gradient descent 
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        # Run  
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Using mini-batch training 
        def _feed_dict(batch_size):
            indices = np.random.choice(64, batch_size)
            sample_images = images[indices,...]
            sample_labels = labels[indices, ...]
            return {x: sample_images, y: sample_labels} #, batch_size_holder:batch_size}
        accuracy_set = [] 
        loss_set = [] 
        train_flag = True

        for epoch in range(num_epoches):
            if train_flag == False:
                break 
            for iter in range(num_iters_per_epoch):
                # Train 
                _ = sess.run([train_op],feed_dict=_feed_dict(batch_size))

                # Test 
                y_pred, loss_val, gt, prop_pred, correct_set = sess.run([predict, loss, y, logits, correct_prediction],\
                                                                           feed_dict={x: images, y: labels})
                # Store results 
                train_accuracy = np.mean(correct_set)
                accuracy_set.append(train_accuracy*100)
                loss_set.append(loss_val)
                # Stop if accuracy is good 
                if train_accuracy == 1.0:
                    train_flag = False 
                    break 
#                 print("Epoch = %d, loss = %.2f, train accuracy = %.2f%%") % (epoch + 1, loss_val, 100. * train_accuracy)
        sess.close()
        return accuracy_set, loss_set 


########################################################################
learning_rate = 0.1
batch_size = 64
num_epoches = 2000
num_iters_per_epoch =  5

# Retrieve image data 
images = np.load('./data/random/random_imgs.npy')
labels = np.load('./data/random/random_labs.npy')
 
labels = np.expand_dims(labels,1)
labels = np.concatenate((labels, 1 - labels), axis=1).astype(np.float32)


## 
FNN_ins = FNN('sigmoid_activation', 'sigmoid_L2')
accuracy_set, loss_set = FNN_ins.train(images, labels) 

draw_accu_loss(accuracy_set, loss_set)

## 
FNN_ins = FNN('sigmoid_activation', 'sigmoid_cross_entropy')
accuracy_set, loss_set = FNN_ins.train(images, labels) 

draw_accu_loss(accuracy_set, loss_set)

## 
FNN_ins = FNN('relu_activation', 'sigmoid_L2')
accuracy_set, loss_set = FNN_ins.train(images, labels) 

draw_accu_loss(accuracy_set, loss_set)

## 
FNN_ins = FNN('relu_activation', 'sigmoid_cross_entropy')
accuracy_set, loss_set = FNN_ins.train(images, labels) 

draw_accu_loss(accuracy_set, loss_set)

## 
