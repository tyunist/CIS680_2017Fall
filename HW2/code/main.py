import numpy as np                                                                                    
import tensorflow as tf
import os 
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
 
  train_data_loader, train_label_loader = get_loader(
    config.data_path, config.batch_size, 'train', True)

  if config.is_train:
    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test, 'test', False)
  else:
    test_data_loader, test_label_loader = get_loader(
      config.data_path, config.batch_size_test, config.split, False)
  print('\n.....Start training')
  trainer = Trainer(config, train_data_loader, train_label_loader, test_data_loader, test_label_loader)
  if config.is_train:
    print('\n.....Training mode')
    save_config(config)
    train_error_set = trainer.train()
  else:
    print('\n.....Test mode')
    if not config.load_path:
      raise Exception("[!] You should specify `load_path` to load a pretrained model")
    trainer.test()

  # Draw training error 
  np.savetxt(os.path.join(config.log_dir,'train_accuracy.txt'), train_error_set)
  fig = plt.figure()
  plt.plot(train_error_set[:,2])
  plt.title('Training Acurracy over Iterations')
  plt.xlabel('Iterations')
  plt.ylabel('Accuracy')
  fig.savefig(os.path.join(config.log_dir,'training_accuracy.png'))

  print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

def run():
  config, unparsed = get_config()
  main(config)

if __name__ == "__main__":
  run()
