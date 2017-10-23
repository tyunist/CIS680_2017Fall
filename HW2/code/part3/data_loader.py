import os, pdb 
import numpy as np
import tensorflow as tf
import dataset_utils
import cv2 
slim = tf.contrib.slim

IMG_MEAN = 121.285
IMG_STD = 64.226

def download_data(data_path, unpack=False):
  url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  if not tf.gfile.Exists(data_path):
    tf.gfile.MakeDirs(data_path)
    dataset_utils.download_and_uncompress_tarball(url, data_path)
  print os.path.join(data_path , "imgs") 
  print tf.gfile.Exists(os.path.join(data_path , "imgs"))
  if tf.gfile.Exists(os.path.join(data_path , "imgs")) and unpack==False:
    if len(os.listdir(os.path.join(data_path , "imgs"))) > 100:
      print('>>Data are already unpackaged to', os.path.join(data_path ,"imgs"))
      return 
  print('...Unpacking data from ', data_path)
  dataset_utils.unpack_cifar10(data_path , PIXELS_DIR = "imgs")
  print('>>Complete unpackaging to', os.path.join(data_path , "imgs"))

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
    img_paths.append(img_dir + img_name)
    labs.append(int(lab)) 
  f.close()
  return img_paths, labs

def read_images_from_disk(input_queue):
  """Consumes a single filename and label as a ' '-delimited string
  Args:
    filename_and_label_tensor: A scalar string tensor
  Returns:
    Two tensors: the decoded image, and the string label
  """
  lab = input_queue[1]
  img_path = tf.read_file(input_queue[0])
  img = tf.image.decode_png(img_path, channels=3)
  return img, lab

def get_loader(root, batch_size, mode_list=[None], split=None, shuffle=True):
  """ Get a data loader for tensorflow computation graph
  Args:
    root: Path/to/dataset/root/, a string
    batch_size: Batch size, a integer
    split: Data for train/val/test, a string
    shuffle: If the data should be shuffled every epoch, a boolean
  Returns:
    img_batch: A (float) tensor containing a batch of images.
    lab_batch: A (int) tensor containing a batch of labels.
  """
  img_paths_np, labs_np = read_labeled_image_list(root+ '/' + split+'.txt', root+'/imgs/')

  with tf.device('/cpu:1'):
    img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)
    labs = tf.convert_to_tensor(labs_np, dtype=tf.int64)

    input_queue = tf.train.slice_input_producer([img_paths, labs],
                  shuffle=shuffle, capacity=10*batch_size)

    img, lab = read_images_from_disk(input_queue)

    img.set_shape([32, 32, 3])
    img = tf.cast(img, tf.float32)

    # Preprocessing images 
    if not mode_list[0]:
      print '...Mode: not preprocessing'
   
    else: 
      # Augmenting in order 
      if 'flip_horizontal' in mode_list and split=='train': # only flip when training 
        img = tf.image.random_flip_up_down(img)
        print '...Mode: flip_horizontal'

    img_batch, lab_batch = tf.train.batch([img, lab], num_threads=1,
                           batch_size=batch_size, capacity=10*batch_size)
    if mode_list[0]:
      if 'normalize' in mode_list:
        print '...Mode: normalize'
        # vs = tf.get_variable_scope()
        mean, var = tf.nn.moments(img_batch, axes=[0,1,2], keep_dims=True)
        img_batch = (img_batch - mean)/tf.sqrt(var)
        # img_batch = slim.batch_norm(img_batch, is_training=False, reuse=None, scale=True, scope=vs)
      if 'pad_crop' in mode_list and split=='train':
          print '...Mode: zero pad and cropping'
          # padding 
          paddings = tf.constant([[0,0], [4, 4], [4, 4], [0, 0]]) # pad 4 elements to each dimension 
          padded_img  = tf.pad(img_batch, paddings, 'CONSTANT')

          # Randomly crop 
          img_batch = tf.random_crop(padded_img, tf.constant([batch_size,32,32,3]))

  return img_batch, lab_batch



def np_get_some_test_images(root, num_img_per_class, batch_size, mode_list=[None]):
  """ Get a data loader for tensorflow computation graph
  Args:
    root: Path/to/dataset/root/, a string
    batch_size: Batch size, a integer
    split: Data for train/val/test, a string
    shuffle: If the data should be shuffled every epoch, a boolean
  Returns:
    img_batch: A (float) tensor containing a batch of images.
    lab_batch: A (int) tensor containing a batch of labels.
  """
  def _save_some_test_image_each_class_list(img_list_path, img_dir, num_img_per_class, \
                                         correct_label_list_path, wrong_label_list_path):

    """Reads a .txt file containing pathes and labeles
    Args:
      img_list_path: a .txt file with one /path/to/image with one label per line
      img_dir: path of directory that contains images
    Returns:
      List with all filenames
    """
    f = open(img_list_path, 'r')   
    if os.path.exists(correct_label_list_path):
      os.remove(correct_label_list_path)
      os.remove(wrong_label_list_path)
    correct_label_f = open(correct_label_list_path, 'wb')
    wrong_label_f = open(wrong_label_list_path, 'wb')
    remain_img_per_class = num_img_per_class*np.ones(10)
    img_paths = []
    correct_labs = []
    wrong_labs = []
    for line in f:
      img_name, lab = line[:-1].split(' ')
      # append if there is not enough image of label 'lab'
      if remain_img_per_class[int(lab)] > 0:
        img_paths.append(img_dir + img_name)
        correct_labs.append(int(lab)) 
        wrong_labs.append(9 - int(lab)) 
        remain_img_per_class[int(lab)] -= 1 
        print('>> Add img %s %d'%(img_dir + img_name, int(lab)))
        correct_label_f.write("{0} {1}\n".format(img_dir + img_name, int(lab)))
        wrong_label_f.write("{0} {1}\n".format(img_dir + img_name, 9 - int(lab)))
    f.close()
    correct_label_f.close()
    wrong_label_f.close()
    return img_paths, correct_labs, wrong_labs 
  ##################################################################################################### 
  num_class = 1 
  img_paths_np, correct_labs_np, wrong_labs_np =  _save_some_test_image_each_class_list(root+ '/' +'test.txt', root+'/imgs/',\
                    num_img_per_class, root+ '/' +'20_correct_test.txt', root+ '/' +'20_wrong_test.txt')
  
  img_batch_np = np.zeros([num_img_per_class*num_class, 32, 32, 3]) 
  correct_labs_np = np.array(correct_labs_np)
  wrong_labs_np = np.array(wrong_labs_np) 
  for i in range(num_img_per_class*num_class):
    img_batch_np[i] = cv2.imread(img_paths_np[i])
    print 'read', img_paths_np[i]
      
  return img_batch_np, correct_labs_np[0:num_class*num_img_per_class], wrong_labs_np[0:num_class*num_img_per_class]



def np_get_one_test_images(root, img_number, class_id,  mode_list=[None]):
  """  
  Args:
    root: Path/to/dataset/root/, a string
  Returns:
    img: img
    lab: label 
  """
  img_list_path = root+ '/' +'test.txt' 
  print 'index:', img_number 
  f = open(img_list_path, 'r')
  row = 0    
  for line in f:
    img_name, lab = line[:-1].split(' ')
    row += 1 
    if row < img_number:
      continue
    if int(lab) == class_id:
      # image not read 
 
      img_path = os.path.join(root + '/imgs/', img_name) 
      print('Img path:', img_path)
      img_np  = cv2.imread(img_path)

      if not mode_list[0]:
        pass 
        # print '...Mode: not preprocessing'
      else:
        if 'normalize' in mode_list:
          print '...Mode: normalize'
          img_np = (img_np - IMG_MEAN)/IMG_STD 
      return img_np, class_id, row 

  return None 
