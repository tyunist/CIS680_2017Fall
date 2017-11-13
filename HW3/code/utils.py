'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
from torchvision import transforms, utils
import matplotlib.pyplot as plt 
import pdb
import numpy as np
from skimage.draw import polygon 
from skimage.transform import resize 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

def draw_rectangle(box, img):
  img_size = img.shape[0] 
  center_y, center_x, w = box[0], box[1], box[2]
  y_start = max(0, center_y - w/2) 
  x_start = max(0, center_x - w/2) 
  y_end = min(img_size, y_start + w)
  x_end = min(img_size, x_start + w) 
  start, end = (y_start, x_start), (y_end, x_end) 
  
  rows = np.array([y_start, y_start, y_end, y_end])
  cols = np.array([x_start, x_end, x_end, x_start])
  rr, cc = polygon(rows,cols)  
  img[rr, cc, 0] = 10 
  img[rr, cc, 1] = 5 
  img[rr, cc, 2] = 10 
  return img 

def batch_display(sample_batched):
  images_batch, label_batch, box_batch, mask_batch = sample_batched['image'], sample_batched['label'], sample_batched['box'], sample_batched['mask']
  batch_size = len(images_batch) 
      
  img = images_batch[0].numpy().transpose((1,2,0))
  rgb = np.fliplr(img.reshape(-1,3)).reshape(img.shape)
  mask = mask_batch[0].numpy() 
  print('size of mask:', mask.shape) 
  img_size = img.shape[0] 
  # Draw bounding box 
  box = img_size*box_batch[0][0].numpy()
  print('box:', box)  
  
  boxed_img = draw_rectangle(box, rgb.copy()) 

  # Draw mask 
  resized_img = resize(rgb.astype(np.int8), (img_size/8.0, img_size/8.0)) 
  print('New image size:', resized_img.shape)
  one_mask = np.where(mask==1) 
  zero_mask = np.where(mask==0)
  
  resized_img[one_mask[0], one_mask[1],:] = np.array([100, 100, 20])  
  resized_img[zero_mask[0], zero_mask[1],:] = 0 
  
  plt.subplot(2,2,1)
  plt.imshow(rgb)
  plt.title(classes[label_batch[0].numpy()[0]])  
  plt.subplot(2,2,2)
  plt.imshow(mask,cmap='gray') 
  plt.title('Mask') 
  plt.subplot(2,2,3)
  plt.imshow(boxed_img) 
  plt.title('Bounding box') 
  plt.subplot(2,2,4)
  plt.imshow(resized_img) 
  plt.title('Mask box') 
  #grid = utils.make_grid(images_batch)
  #plt.imshow(grid.numpy().transpose((1,2,0)))
  #plt.imshow(de_norm(images_batch[1].numpy().transpose((1,2,0))))

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

    
def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    return tot_time  

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
