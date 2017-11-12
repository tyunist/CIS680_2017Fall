import torch
import torch.utils.data as data 
from torch.autograd import Variable  
from six.moves import urllib
import glob
import numpy as np
from skimage.io import imsave
import pickle as pkl
import pdb, os, sys  
import tarfile
from skimage import io, transform 
import matplotlib.pyplot as plt  
import numbers 


class Cifar10_transformed_ToTensor(object):
  """Convert ndarrays in sample to Tensors."""
  def __call__(self, sample):
    image, label, box, mask = sample['image'], sample['label'], sample['box'], sample['mask']

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return {'image': torch.from_numpy(image).float(),
            #'label': torch.LongTensor(label) }
            'label': torch.from_numpy(label).type(torch.LongTensor),
            'box': torch.from_numpy(box).float(),
            'mask': torch.from_numpy(mask).float()}



class cifar10_transformed_loader_obj(data.Dataset):
  def __init__(self, param, transform=None):
    self.data, self.label, self.box, self.mask = read_cifar10_transformed_list(param['label_path'], param['data_path'], param['mask_path'])
    self.transform = transform # this can be a compose of transforms 


  def __getitem__(self, index):
    #img = np.array(Image.open(self.data[index]))[np.newaxis,...]
    img = io.imread(self.data[index])
    mask = io.imread(self.mask[index])
    label = np.array([self.label[index]]) 
    box = np.array([self.box[index]])
    #label = self.label[index].astype(np.int) 
    #img = torch.from_numpy(img.astype(np.float32))
    #label = torch.from_numpy(label)
    sample = {'image':img, 'label': label, 'box': box, 'mask':mask} 
    if self.transform:
      sample = self.transform(sample)
    return sample 
    
  def __len__(self):
    return self.data.shape[0] 


def read_cifar10_transformed_list(img_list_path, img_dir, mask_dir, image_size=48):
  """Reads a .txt file containing pathes and labeles
  Args:
    img_list_path: a .txt file with one /path/to/image with one label per line
    img_dir: path of directory that contains images
    mask_dir: path of directory that contains masks
  Returns:
    List with all filenames
  """
  f = open(img_list_path, 'r')
  img_paths = []
  labs = []
  box  = [] 
  mask_paths = [] 
  for line in f:
    img_name, lab, row, col, width = line[:-1].split(' ')
    img_paths.append(os.path.join(img_dir, img_name))
    mask_paths.append(os.path.join(mask_dir, img_name))
    labs.append(int(lab)) 
    # here, boxes are squares. We want to scale it with respect to the image size 
    box.append([float(row)/image_size, float(col)/image_size, float(width)/image_size]) 
  f.close()
  return np.array(img_paths), np.array(labs), np.array(box), np.array(mask_paths) 


