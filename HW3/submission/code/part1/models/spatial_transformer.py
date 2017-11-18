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

def box_proposal_to_theta(boxes):
  """Transform the output of box regression to affine tranformations
     Input:   boxes tensor(Variable) size N x 3 
     Outputs: theta size N x 2 x 3"""
  batch_size = boxes.size(0)
  theta = torch.cat((boxes, boxes), 1)
  theta[:,0] = boxes[:,2]/48.0
  theta[:,1] = 0
  theta[:,2] = (boxes[:,1] - 24.0)/24.0
  theta[:,3] = 0
  theta[:,4] = boxes[:,2]/48.0
  theta[:,5] = (boxes[:,0] - 24.0)/24.0
  return theta  
  

def torch_spatial_transformer(img, theta, output_size=None):
  """Spatial transformer
     Inputs: img  tensor size: N x H x W 
             theta tensor size       N x 2 x 3 
             output_size tensor (N, 3, H, W). None means output size  = input size"""
  if output_size is None:
    output_size = img.size()
  if not isinstance(output_size, torch.Size) and len(output_size)  == 2: 
    batch_size = img.size(0)
    output_size = torch.Size([batch_size, 3, output_size[0], output_size[1]])   
  if theta.ndimension()  < 3:
    theta = theta.view(-1, 2, 3) 
  grid = F.affine_grid(theta, output_size)
  transformed_img = F.grid_sample(img, grid) 
  return transformed_img 
