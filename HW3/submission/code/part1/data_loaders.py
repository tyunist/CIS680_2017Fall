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

class RandomCrop(object):
  def __init__(self, size, padding=0):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size 
    self.padding = padding 

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    if self.padding > 0:
      image = np.pad(image, [(self.padding, self.padding), (self.padding, self.padding), (0, 0)], mode='constant')
    w, h = image.shape[1], image.shape[0]
    th, tw = self.size 
    if w==tw and h==th:
      pass
    else:
      x1 = np.random.randint(0, w-tw)
      y1 = np.random.randint(0, h-th)
      image = image[y1: y1+th, x1: x1+tw, :]
    return {'image':image,  
            'label':label}   
  

class RandomFlipHorizontal(object):
  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    if np.random.random() < 0.5:
      for i in range(3):
        image[:,:,i] = np.fliplr(image[:,:,i]).T
    return {'image':image,  
            'label':label}   

class ToTensor(object):
  """Convert ndarrays in sample to Tensors."""
  def __call__(self, sample):
    image, label = sample['image'], sample['label']

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image = image.transpose((2, 0, 1))
    return {'image': torch.from_numpy(image).float(),
            #'label': torch.LongTensor(label) }
            'label': torch.from_numpy(label).type(torch.LongTensor)}

class Normalize(object):
  """Normalize images"""
  # NOTE: change image pixels to float before standardization. Otherwise, it does not work! 
  def __init__(self, mean, std):
    self.mean = mean 
    self.std = std 
  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    #image = (image - self.mean)/self.std
    in_channels = len(self.mean) 
    assert len(image.shape) == in_channels, 'ERROR <Normalize>: in_channels must equal len(image.shape)'
    if in_channels == 1:
      image = (image - self.mean[0])/self.std[0]
    else:
      img_shape = image.shape 
      if len(img_shape) == 3:
        for i in range(in_channels):
          image[:,:,i] = (image[:,:,i] - self.mean[i])/self.std[i]
      if len(img_shape) == 4:
        for i in range(in_channels):
          image[:,:,:,i] = (image[:,:,:,i] - self.mean[i])/self.std[i]
        
    return {'image': image,
            'label': label}
class Rescale(object):
  """Rescale the image in a sample to a given size.

  Args:
      output_size (tuple or tuple): Desired output size. If tuple, output is
          matched to output_size. If int, smaller of image edges is matched
          to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, sample):
    image, label = sample['image'], sample['label']
    h, w = image.shape[:2]
    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)
    
    # This function will scale image by 255. 
    img = transform.resize(image, (new_h, new_w))
    return {'image': img, 'label': label}


class dataloader_obj(data.Dataset):
  def __init__(self, param, transform=None):
    self.data, self.label = read_labeled_image_list(param['label_path'], param['data_path'])
    self.transform = transform # this can be a compose of transforms 


  def __getitem__(self, index):
    #img = np.array(Image.open(self.data[index]))[np.newaxis,...]
    img = io.imread(self.data[index])
    label = np.array([self.label[index]]) 
    #label = self.label[index].astype(np.int) 
    #img = torch.from_numpy(img.astype(np.float32))
    #label = torch.from_numpy(label)
    sample = {'image':img, 'label': label} 
    if self.transform:
      sample = self.transform(sample)
    return sample 
    
  def __len__(self):
    return self.data.shape[0] 



class rawData_prepare():
  """This class executes all steps in raw data  preprocessing including: download, 
     convert to pickle...."""
  def __init__(self, raw_data_path):
    self.imgs = os.path.join(raw_data_path, 'imgs') # path to images 
  def download_tar(self, raw_data_path, url=None, download=False):
    if not url:
      url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    if not os.path.exists(raw_data_path):
      os.makedirs(raw_data_path)
    print('===> Download to ', raw_data_path)
    if download:
      download_and_uncompress_tarball(url, raw_data_path)
    if os.path.exists(self.imgs) and len(os.listdir(self.imgs)) > 100:
        print('===> Data are already unpacked to', self.imgs)
        save_img = False 
    else:
      save_img = True 
    print('===> Unpacking data from ', raw_data_path)
    self.mean, self.std = unpack_cifar10(raw_data_path, save_img)
    return self.mean, self.std  



##################################################################################
def read_labeled_image_list(img_list_path, img_dir):
  """Reads a .txt file containing pathes and labeles
  Args:
    img_list_path: a .txt file with one /path/to/image with one label per line
    img_dir: path of directory that contains images
  Returns:
    List with all filenames
  """
  f = open(img_list_path, 'r')
  img_paths = []
  labs = []
  for line in f:
    img_name, lab = line[:-1].split(' ')
    img_paths.append(os.path.join(img_dir, img_name))
    labs.append(int(lab)) 
  f.close()
  return np.array(img_paths), np.array(labs)


def unpack_cifar10(raw_path, save_img=True):
  """
      Entry point.
  """
  
  sum_pixel_intensity = np.zeros((3,1024*20000)) 
  train_label_file = "train.txt"
  test_label_file = "test.txt"
  
  PIXELS_DIR = os.path.join(raw_path, 'imgs')
  train_label_file = os.path.join(raw_path, train_label_file)
  test_label_file = os.path.join(raw_path, test_label_file)
  if not os.path.exists(PIXELS_DIR):
    os.makedirs(PIXELS_DIR) 

  def unpack_file(fname):
    """
        Unpacks a CIFAR-10 file.
    """
    with open(fname, "rb") as f:
        result = pkl.load(f, encoding='bytes') # encoding to work with python3
    return result
  def save_as_image(img_flat, fname):
    """
        Saves a data blob as an image file.
    """

    # consecutive 1024 entries store color channels of 32x32 image 
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    print('save ', os.path.join(PIXELS_DIR, fname))
    imsave(os.path.join(PIXELS_DIR, fname), img)

  train_labels = {}  
  test_labels =  {} 
  pdb.set_trace()
  # use "data_batch_*" for just the training set
  for fname in glob.glob(os.path.join(raw_path,'cifar-10-batches-py') +"/data_batch_1"):
    data = unpack_file(fname)
    for i in range(10000):
      img_flat = data[b"data"][i]
      fname = data[b"filenames"][i].decode('utf-8')
      label = data[b"labels"][i]

      # save the image and store the label
      if save_img:
        save_as_image(img_flat, fname)
      train_labels[fname] = label 
      # # Accumulate sum_intensity 
      sum_pixel_intensity[0,i*1024:(i+1)*1024]= img_flat[0:1024]
      sum_pixel_intensity[1,i*1024:(i+1)*1024]= img_flat[1024:2048]
      sum_pixel_intensity[2,i*1024:(i+1)*1024]= img_flat[2048:3072]


  # write out labels file
  with open(train_label_file, "w") as f:
    for (fname, label) in train_labels.items():
       f.write("{0} {1}\n".format(fname, label)) 
  
  # use "test_batch*" for just the test set
  for fname in glob.glob(os.path.join(raw_path,'cifar-10-batches-py') +"/test_batch*"):
    data = unpack_file(fname)
    for i in range(10000):
      img_flat = data[b"data"][i]
      fname = data[b"filenames"][i].decode('utf-8')
      label = data[b"labels"][i]
      # save the image and store the label
      if save_img:
        save_as_image(img_flat, fname)
      test_labels[fname] = label
      sum_pixel_intensity[0,(10000+i)*1024:(i+10001)*1024]= img_flat[0:1024]
      sum_pixel_intensity[1,(10000+i)*1024:(i+10001)*1024]= img_flat[1024:2048]
      sum_pixel_intensity[2,(10000+i)*1024:(i+10001)*1024]= img_flat[2048:3072]
          
  # write out labels file
  with open(test_label_file, "w") as f:
    for (fname, label) in test_labels.items():
      f.write("{0} {1}\n".format(fname, label))
  # Compute mean and var of intensity 
  print('Max, min', np.max(sum_pixel_intensity, axis=1), np.min(sum_pixel_intensity, axis=1))
  mean =  np.mean(sum_pixel_intensity, axis=1)
  std  =  np.std(sum_pixel_intensity, axis=1)
  print ('Mean:', mean) 
  print ('std:', std)
  return mean, std  


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.
  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)
