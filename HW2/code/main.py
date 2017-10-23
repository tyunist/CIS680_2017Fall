import numpy as np
import pdb                                                                                      
import tensorflow as tf
import os, shutil
from trainer import Trainer
from config import get_config
from data_loader import get_loader, download_data, get_some_test_images_loader, np_get_some_test_images 
from utils import prepare_dirs_and_logger, save_config
import matplotlib.pyplot as plt 
def main(config):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  # Download data 
  download_data(config.data_path)
  print('>>Complete downloading data')
  
 

 
  print('...*****************Training mode*************************...')
  print('\n.....Obtain train & test batches')
  # Load 20 images for the first time 
  train_data_loader, train_correct_label_loader, train_wrong_label_loader = np_get_some_test_images(config.data_path, 1, \
                                                config.batch_size, config.preprocessing_list)
  pdb.set_trace() 
  if not config.use_pretrained:
    print('.....Deleting the current model in:', os.path.join(config.model_dir, '*'))
    try:
      shutil.rmtree(config.model_dir)
    except:
      pass 
  print('\n.....Start training')
  trainer = Trainer(config, train_data_loader, train_correct_label_loader, train_wrong_label_loader)

  

  if config.is_train:
    save_config(config)
    train_error_set = trainer.train()
    #   # Draw training error 
    # np.savetxt(os.path.join(config.log_dir,'train_accuracy.txt'), train_error_set)
    # fig = plt.figure()
    # plt.plot(train_error_set[:,2])
    # plt.title('Training Acurracy over Iterations')
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # fig.savefig(os.path.join(config.log_dir,'training_accuracy.png'))

    # if config.get_cnn_grad:
    #   print('...Get CNN gradients')
    #   fig = plt.figure()
    #   plt.plot(train_error_set[:,3])
    #   plt.title('Gradient in The First CNN Layer over Iterations')
    #   plt.xlabel('Iterations')
    #   plt.ylabel('Gradient')
    #   fig.savefig(os.path.join(config.log_dir,'conv1_grad.png'))

    #   fig = plt.figure()
    #   plt.plot(train_error_set[:,4])
    #   plt.title('Gradient in The Last CNN Layer over Iterations')
    #   plt.xlabel('Iterations')
    #   plt.ylabel('Gradient')
    #   fig.savefig(os.path.join(config.log_dir,'conv4_grad.png'))

  else:
    print('\n.....Test mode')
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()



  print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

def run():
  config, unparsed = get_config()
  main(config)

if __name__ == "__main__":
  run()
