import numpy as np
import pdb                                                                                      
import tensorflow as tf
import os, shutil, cv2 
from trainer import Trainer
from config import get_config
from data_loader import get_loader, download_data, np_get_some_test_images, np_get_one_test_images
from utils import prepare_dirs_and_logger, save_config
import matplotlib.pyplot as plt 
IMG_MEAN = 121.285
IMG_STD = 64.226
def un_normalize(img_np):
  return np.uint8(img_np*IMG_STD + IMG_MEAN)
def main(config, class_id=1, adv_img_number=1, index=0):
  tf.reset_default_graph()
  result_path = config.adv_dir
  if not os.path.exists(result_path):
    os.makedirs(result_path)
  if not os.path.exists(result_path + 'imgs/'): 
    os.makedirs(result_path + 'imgs/')
    os.makedirs(result_path + 'figs/')

  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  # Download data 
  download_data(config.data_path)
  print('>>Complete downloading data')

 
  print('...*****************Training mode*************************...')
  print('\n.....Obtain one image, class ', class_id)

  trainer = Trainer(config, None, None, None)
  # img_numb = adv_img_number 
 
  while True:
    img_np, class_id, row   = np_get_one_test_images(config.data_path, index, class_id,\
                                        mode_list=config.preprocessing_list)
    index = row + 1 
    # pdb.set_trace() 


    # Test whether image is valid 
    test_confidence = trainer.test_valid_sample(img_np, class_id )
    if test_confidence[0,class_id] == np.max(test_confidence[0]) and test_confidence[0,class_id] < 0.3 :
      print'>>> Choose this image with confidence %f'%(test_confidence[0,class_id])
      # Choose wrong class ID 
      test_confidence = test_confidence[0]
      test_confidence[class_id] = 0 
      wrong_class_id = np.argmax(test_confidence)
      print'>>> Choose wrong ID %d confidence %f'%(wrong_class_id, test_confidence[wrong_class_id])
      # plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
      # plt.show()
      break 
  print'\n****  Test image number %d, class %d'%(index, class_id)
  print 'index:', index 
  tf.reset_default_graph()
  # Train the image 
  trainer = Trainer(config, img_np, class_id, wrong_class_id)
  adv_img_name = os.path.join(result_path + 'imgs/', str(class_id) + '_' + str(adv_img_number) + '.jpg')
  result_fig_name = os.path.join(result_path + 'figs/', str(class_id) + '_' + str(adv_img_number) + '.jpg')
 
  adv_img_f =  open(config.adv_list, 'ab')
  trainer.train_adv_image(adv_img_number, adv_img_name, adv_img_f, result_fig_name)
  adv_img_f.close()


  print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  return index  

def run():
  config, unparsed = get_config()
  if os.path.exists(config.adv_dir):
    shutil.rmtree(config.adv_dir)
  for class_id in range(10):
    index = 0 
    for adv_img_number in range(1,3):
      index = main(config, class_id, adv_img_number, index)

if __name__ == "__main__":
  run()
