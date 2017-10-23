import numpy as np                                                                                    
import tensorflow as tf
import os, shutil
from trainer import Trainer
from config import get_config
from data_loader import get_loader, download_data 
from utils import prepare_dirs_and_logger, save_config
import matplotlib.pyplot as plt 
def main(config):
  prepare_dirs_and_logger(config)

  rng = np.random.RandomState(config.random_seed)
  tf.set_random_seed(config.random_seed)

  # Download data 
  download_data(config.data_path)
  print('>>Complete downloading data')
  print('\n.....Obtain train & test batches')
  print('\n.....Model_dir:', config.model_dir) 

 
  train_data_loader, train_label_loader = get_loader(
    config.data_path, config.batch_size, config.preprocessing_list, 'train', True)

  if config.is_train:
    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test, config.preprocessing_list, 'test', False)
  else:
    print('...Testing mode. Currently, data used to test is', config.split)
    print('...Testing mode. What data you want to test on? ')
    command = raw_input('press y or yes to do test with test data. Else, evaluate on train data:')
    if command == 'y' or command == 'yes' or command == 'Y':
      config.split = 'test'
    else: 
      config.split = 'train'
    print('...Testing on %s data'%config.split)

    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test, config.preprocessing_list, config.split, False)
  print('\n.....Training mode')
  
  if not config.use_pretrained:
    print('.....Deleting the current model in:', os.path.join(config.model_dir, '*'))
    try:
      shutil.rmtree(config.model_dir)
    except:
      pass 
  print('\n.....Start training')
  trainer = Trainer(config, train_data_loader, train_label_loader, test_data_loader, test_label_loader)
  if config.is_train:
    save_config(config)
    train_error_set = trainer.train()
      # Draw training error 
    np.savetxt(os.path.join(config.log_dir,'train_accuracy.txt'), train_error_set)
    fig = plt.figure()
    plt.plot(train_error_set[:,2])
    plt.title('Training Acurracy over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    fig.savefig(os.path.join(config.log_dir,'training_accuracy.png'))

    if config.get_cnn_grad:
      print('...Get CNN gradients')
      fig = plt.figure()
      plt.plot(train_error_set[:,3])
      plt.title('Gradient in The First CNN Layer over Iterations')
      plt.xlabel('Iterations')
      plt.ylabel('Gradient')
      fig.savefig(os.path.join(config.log_dir,'conv1_grad.png'))

      fig = plt.figure()
      plt.plot(train_error_set[:,4])
      plt.title('Gradient in The Last CNN Layer over Iterations')
      plt.xlabel('Iterations')
      plt.ylabel('Gradient')
      fig.savefig(os.path.join(config.log_dir,'conv4_grad.png'))

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
