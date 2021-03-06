#-*- coding: utf-8 -*-                                                                                
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--c_num', type=int, default=10)  # Number of classes

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
data_arg.add_argument('--split', type=str, default='train')
data_arg.add_argument('--batch_size', type=int, default=100)
data_arg.add_argument('--batch_size_test', type=int, default=20)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--use_pretrained', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=2000)
train_arg.add_argument('--epoch_step', type=int, default=50) # v12 = 50, original 100 
train_arg.add_argument('--lr', type=float, default=1e-3) # 1e-4: to resolve gradient vanishing using 1 res. Original: 1e-3 
train_arg.add_argument('--min_lr', type=float, default=1e-4)
train_arg.add_argument('--wd_ratio', type=float, default=5e-2)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
main_path = '/home/tynguyen/'
#main_path = '/media/sf_cogntivive_school/' 
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=100)
misc_arg.add_argument('--test_iter', type=int, default=100)
misc_arg.add_argument('--save_step', type=int, default=100)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])

# log_sub_path = 'customized_cnn_resolve_vanishing' 
log_sub_path = 'customized_cnn_normalize' 
#log_sub_path = 'quick_cnn_normalize' 
misc_arg.add_argument('--load_path', type=str, default= main_path + 'cis680/logs/HW2/' + log_sub_path)
misc_arg.add_argument('--log_dir', type=str, default= main_path + 'cis680/logs/HW2/' + log_sub_path)
misc_arg.add_argument('--data_dir', type=str, default= main_path + 'cis680/data/')
misc_arg.add_argument('--adv_dir', type=str, default= main_path + 'cis680/data/results/adv/')
misc_arg.add_argument('--adv_list', type=str, default= main_path + 'cis680/data/results/adv/advs.txt') 
misc_arg.add_argument('--random_seed', type=int, default=0)

# Question 1.1, 1.2: choose normalizing or not 
misc_arg.add_argument('--preprocessing_list', nargs='+',default=  ['normalize']) # set default = [None] to do nothing 

# Question 2.1 
#misc_arg.add_argument('--cnn_model', type=str, default='quick_cnn')
misc_arg.add_argument('--cnn_model', type=str, default='customized_cnn')
# Question 2.2 
misc_arg.add_argument('--get_cnn_grad', type=str2bool, default=True)
# Question 2.3 
misc_arg.add_argument('--make_grad_vanish', type=str2bool, default=False) # Default should be False 
# Question 2.4: resolve gradient vainishing 
misc_arg.add_argument('--resolve_grad_vanish', type=str2bool, default=False) # Default should be False 

def get_config():
  config, unparsed = parser.parse_known_args()
  return config, unparsed    

if __name__ == '__main__':
	config, unparsed  = get_config()
	print 'config:', config 
	print 'unparse:', unparsed 
