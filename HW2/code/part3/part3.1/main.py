import numpy as np
import pdb                                                                                      
import tensorflow as tf
import os, shutil, cv2 
from trainer import Trainer
from config import get_config
from data_loader import get_loader, download_data, np_get_one_test_images
from utils import prepare_dirs_and_logger, save_config
import matplotlib.pyplot as plt 
import time 
IMG_MEAN = 121.285
IMG_STD = 64.226
def un_normalize(img_np):
  return np.uint8(img_np*IMG_STD + IMG_MEAN)
def main(config, class_id=1):
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
  batch_size = config.batch_size 
  img_np = np.zeros([batch_size, 32, 32, 3])
  correct_labels_np = np.zeros([batch_size])
  wrong_labels_np = np.zeros([batch_size])
  original_confidence_np = np.zeros([batch_size,10])
  wrong_class_id = class_id
  while wrong_class_id == class_id:
    wrong_class_id = np.random.choice(10)

  num_valid_img = 0 
  index = 0 
  while num_valid_img < batch_size:
    single_img_np, class_id, row   = np_get_one_test_images(config.data_path, index, class_id,\
                                        mode_list=config.preprocessing_list)
    index = row + 1 
    # pdb.set_trace() 


    # Test whether image is valid 
    test_confidence = trainer.test_valid_sample(single_img_np, class_id )
    if test_confidence[0,class_id] == np.max(test_confidence[0])   :
      print'>>> Choose this image with confidence %f'%(test_confidence[0,class_id])
      # Choose wrong class ID 
       
      # plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
      # plt.show()

      original_confidence_np[num_valid_img] = test_confidence
      img_np[num_valid_img] = single_img_np 
      correct_labels_np[num_valid_img] = class_id
      wrong_labels_np[num_valid_img] = wrong_class_id 
      num_valid_img += 1 
  
  for adv_img_number in range(1,3):
   
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print'\n****  Test image number %d, class %d'%(index, class_id)
  
    
    tf.reset_default_graph()
    # Train the image 
    trainer = Trainer(config, img_np, correct_labels_np.astype(np.int64), wrong_labels_np.astype(np.int64),original_confidence_np, adv_img_number, test_valid=False)
  
    adv_img_name = os.path.join(result_path + 'imgs/', str(class_id) + '_' + str(adv_img_number) + '.jpg')
    result_fig_name = os.path.join(result_path + 'figs/', str(class_id) + '_' + str(adv_img_number) + '.jpg')
   
    adv_img_f =  open(config.adv_list, 'ab')
    chosen_adv_index = trainer.train_adv_image(adv_img_number, adv_img_name, adv_img_f, result_fig_name)
    adv_img_f.close()

    # Remove the image from the list 
    try:
      original_confidence_np[chosen_adv_index] = original_confidence_np[chosen_adv_index+1]
      img_np[chosen_adv_index] = img_np[chosen_adv_index+1]
      correct_labels_np[chosen_adv_index] = correct_labels_np[chosen_adv_index+1]
      wrong_labels_np[chosen_adv_index] = wrong_labels_np[chosen_adv_index+1]
    except:
      original_confidence_np[chosen_adv_index] = original_confidence_np[chosen_adv_index-1]
      img_np[chosen_adv_index] = img_np[chosen_adv_index-1]
      correct_labels_np[chosen_adv_index] = correct_labels_np[chosen_adv_index-1]
      wrong_labels_np[chosen_adv_index] = wrong_labels_np[chosen_adv_index-1]

    wrong_class_id = class_id
    while wrong_class_id == class_id:
      wrong_class_id = np.random.choice(10)
    for i in range(wrong_labels_np.shape[0]):
      wrong_labels_np[i] = wrong_class_id
  print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  return index  

def run():
  config, unparsed = get_config()
  if os.path.exists(config.adv_dir):
    shutil.rmtree(config.adv_dir)
  for class_id in range(10):
    main(config, class_id)

if __name__ == "__main__":
  run()
