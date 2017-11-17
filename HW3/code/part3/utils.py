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
from torch.autograd import Variable 
import torch.nn.functional as F 
import torch 
import matplotlib.pyplot as plt 
import pdb
import numpy as np
from skimage.draw import polygon 
from skimage.transform import resize 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 


def v2_get_smooth_L1_loss(delta_t):
  """delta_t : N x 3 x d"""
  # Get norm
 
  if  isinstance(delta_t, torch.Tensor):
    delta_t = delta_t.data 
    norm_delta = torch.abs(delta_t) # N x 3 x d , absolute values 
     
 
    # The rest is equivalent to the following naive code 
    # if norm_delta < 1:
    #   return 0.5*norm_delta^2
    # else:
    #   return norm_delta - 0.5 
    one_mask = (norm_delta < 1).float() 
    multi_factor = (1 - torch.mul(one_mask,  torch.FloatTensor([0.5])) )*norm_delta 
    add_factor = torch.mul((1 - one_mask), torch.FloatTensor([-0.5]))

  else:
    norm_delta = torch.abs(delta_t) # N x 3 x d , absolute values 
    one_mask = (norm_delta < 1).float() 
    multi_factor = (1 -  0.5*one_mask)*norm_delta
    add_factor = -0.5*(1 - one_mask) 
  
  return multi_factor*norm_delta + add_factor   

def get_smooth_L1_loss(delta_t):
  """delta_t : N x 3 x d"""
  # Get norm
 
  if  isinstance(delta_t, torch.Tensor):
    delta_t = delta_t.data 
    norm_delta =  torch.norm(delta_t, 2, dim=1) # N x d 
    # The rest is equivalent to the following naive code 
    # if norm_delta < 1:
    #   return 0.5*norm_delta^2
    # else:
    #   return norm_delta - 0.5 
    one_mask = (norm_delta < 1).float() 
    multi_factor = (1 - torch.mul(one_mask,  torch.FloatTensor([0.5])) )*norm_delta 
    add_factor = torch.mul((1 - one_mask), torch.FloatTensor([-0.5]))

  else:
    norm_delta = delta_t.norm(p=2,dim=1)
    one_mask = (norm_delta < 1).float() 
    multi_factor = (1 -  0.5*one_mask)*norm_delta
    add_factor = -0.5*(1 - one_mask) 
  
  return multi_factor*norm_delta + add_factor   



def normalize_reg_outputs(reg, anchor): 
  """Normalize regression outputs
     i.e. tx = (x - x_a)/w_a;   tw = log(w/w_a) 
     Inputs: Variable tensor reg = N x 3 x 36, tensor anchor = N x 3 x 36 
     Outputs: N x 3 x 36 tensor"""
 
  if anchor.ndimension() < 2:
    anchor.view(-1, 1) 
  if reg.ndimension() < 3:
    reg = reg.unsqueeze(2)
    reg = reg.expand_as(anchor)
  x_normed = torch.div(reg[:,0,:] - anchor[:,0,:], anchor[:,2,:])
  y_normed = torch.div(reg[:,1,:] - anchor[:,1,:], anchor[:,2,:])
  w_normed = torch.div(reg[:,2,:],anchor[:,2,:]).log() 
  
  # w_normed = w_normed.expand(xy_normed.size(0), 1, xy_normed.size(2))
  x_normed = x_normed.unsqueeze(1)
  y_normed = y_normed.unsqueeze(1)
  w_normed = w_normed.unsqueeze(1)
  reg = torch.cat((x_normed, y_normed, w_normed ), dim=1)


  # reg[:,0:2,:].data.add_(-anchor[:,0:2,:].data)  # Avoid sharing problem 
  # reg[:,:,:].data.div_(anchor[:,2,:].data)
  # reg[:,2,:].data.log_()
  
  return reg 

def map_mask_2_img_coordinates(mask_size=6, scale=8, start=4):
  """Map coordidates of a mask_size to (2, 36) tensor"""
  X , Y = np.meshgrid(range(mask_size), range(mask_size))
  X = X.reshape(-1)
  Y = Y.reshape(-1)
  indices = np.vstack([Y, X])
  indices = start + scale*indices
  return indices


def get_reg_loss(reg_outputs, boxes, weights, anchors):
  """Get regression loss
    Inputs: reg_outputs : Variable tensor N x 3 x 36
            boxes:        Variable tensor N x 3 
            weights:      Variable tensor N x 36, where only 1 exists 
            - at 1 locations in the mask, else is zero
            anchors:      Variable tensor N x 3"""
  batch_size = reg_outputs.size(0)
  if anchors.ndimension() < 3:
    anchors.unsqueeze(2)
    anchors =  anchors.expand_as(reg_outputs )
  # Normalize reg_outputs 
  norm_reg = normalize_reg_outputs(reg_outputs, anchors)
  # Normalize gt 
  norm_gt = normalize_reg_outputs(boxes, anchors)
  # Obtain Smooth L1 loss
  # TODO: change smooth L1 loss 
  #reg_loss = get_smooth_L1_loss(norm_reg - norm_gt)  # N x 36 
  v2_reg_loss = v2_get_smooth_L1_loss(norm_reg - norm_gt)  # N x 3 x 36 
  
  reg_loss = v2_reg_loss.sum(dim=1) # N x 36 
  
  if (batch_size*weights).data.sum() < 1e-8:
    reg_loss = reg_loss.sum()*0
    print('===> Reg loss = 0 since there is no box') 
    time.sleep(2)
 
  else:
    # reg_loss =  (reg_loss * weights).sum()/(batch_size*weights.sum())
    reg_loss =  (reg_loss * weights).sum()/(weights.sum())
  return reg_loss

def draw_rectangle(box, img):
  img_size = img.shape[0] 
  center_y, center_x, w = box[0], box[1], box[2]
  y_start = min(max(0, center_y - w/2), img_size-1)
  x_start = min(max(0, center_x - w/2), img_size-1)
  y_end = min(img_size-1, y_start + w)
  x_end = min(img_size-1, x_start + w) 
  start, end = (y_start, x_start), (y_end, x_end) 
  
  rows = np.array([y_start, y_start, y_end, y_end])
  cols = np.array([x_start, x_end, x_end, x_start])
  rr, cc = polygon(rows,cols)  
  img[rr, cc, 0] = 10 
  img[rr, cc, 1] = 5 
  img[rr, cc, 2] = 10 
  return img 

def batch_display_transformed(inputs, tf_gt_inputs, tf_pred_inputs=None, num_el=None):
  """Display input images before and after transforming using spatial transformer
    Inputs:  inputs tensor size N x 3 x H x W
             tf_gt_inputs: ground truth of transformed images, tensor size N x 3 x H x W 
             tf_pred_inputs: transformed of predicted images, tensor size N x 3 x H x W """
  # Stack images together
  batch_size = inputs.size(0) 
  if num_el == None:
    num_el = batch_size 
  num_el = min(batch_size, num_el) 
  print('===> Displaying %d pairs..'%num_el) 
  orig_grid = utils.make_grid(inputs.data[:num_el,...], nrow=num_el) # 
  gt_grid = utils.make_grid(tf_gt_inputs.data[:num_el,...], nrow=num_el) # 
  
  if tf_pred_inputs is not None:
    pred_grid = utils.make_grid(tf_pred_inputs.data[:num_el,...], nrow=num_el) # 
  
  # Number of rows in the figure
  num_rows = 2
  if tf_pred_inputs is not None:
    num_rows = 3  
  
  plt.figure(figsize=(2*num_el, 3 * num_rows)) 
  plt.subplot(num_rows, 1, 1)
  plt.imshow(orig_grid.numpy().transpose((1,2,0)).astype(np.uint8))
  plt.title('Original Images')
  plt.axis('off') 
  
  
  plt.subplot(num_rows, 1, 2)
  plt.imshow(gt_grid.numpy().transpose((1,2,0)).astype(np.uint8))
  plt.title('Ground Truth Transformed Images')
  plt.axis('off') 
  
  if tf_pred_inputs is not None:
    plt.subplot(num_rows, 1, 3)
    plt.imshow(pred_grid.numpy().transpose((1,2,0)).astype(np.uint8))
    plt.title('Predicted Transformed Images')
    plt.axis('off') 

def batch_display(sample_batched):
  images_batch, label_batch, box_batch, mask_batch = sample_batched['image'], sample_batched['label'], sample_batched['box'], sample_batched['mask']
  batch_size = len(images_batch) 
      
  #img = images_batch[0].numpy().transpose((1,2,0))
  img = images_batch[0].permute(1,2,0).numpy()
  # TODO: if image display is not RGB,then swap channels: 
  #rgb = np.fliplr(img.reshape(-1,3)).reshape(img.shape)
  rgb = img 
  mask = mask_batch[0].numpy() 
  print('size of mask:', mask.shape) 
  img_size = img.shape[0] 
  # Draw bounding box 
  box = img_size*box_batch[0][0].numpy()
  print('box:', box)  
  
  boxed_img = draw_rectangle(box, rgb.copy()) 

  # Draw mask 
  resized_img = resize(rgb.astype(np.uint8), (img_size/8.0, img_size/8.0)) 
  print('New image size:', resized_img.shape)
  one_mask = np.where(mask==1) 
  zero_mask = np.where(mask==0)
  
  # Make resize_img white 
  resized_img = 255*np.ones_like(resized_img).astype(np.uint8) 
  resized_img[one_mask[0], one_mask[1],:] = np.array([100, 100, 20])  
  resized_img[zero_mask[0], zero_mask[1],:] = 0 
  
  plt.subplot(2,2,1)
  plt.imshow(rgb.astype(np.uint8))
  plt.title(classes[label_batch[0].numpy()[0]])  
  plt.subplot(2,2,2)
  plt.imshow(mask,cmap='gray') 
  plt.title('Mask') 
  plt.subplot(2,2,3)
  plt.imshow(boxed_img.astype(np.uint8)) 
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
import pdb 

def test_get_smooth_L1_loss():
  x = torch.linspace(-10, 10, steps=5).view(-1,1)
  pdb.set_trace()
  num_elements = x.numel()
  y = 2*torch.ones(num_elements).view(-1,1)
  X = torch.cat((x,y), dim=1)
 
  # values = [] 
    
  # for i in range(num_elements):
  #   values.extend([get_smooth_L1_loss(X[i,:])]) 

  x = [v[0] for v in x.numpy()]
  values = get_smooth_L1_loss(X).numpy()
  plt.plot(x, values)
  plt.show()

def test_map_mask_2_img_coordinates():
  a = map_mask_2_img_coordinates()

if __name__ == "__main__":
  test_get_smooth_L1_loss()
  # test_map_mask_2_img_coordinates()
