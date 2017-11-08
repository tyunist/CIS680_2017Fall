import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn 

import torchvision 
import torchvision.transforms as transforms 
from torch.autograd import Variable 
import os 
import argparse 

from models import * 

from utils import progress_bar, batch_display 
from data_loaders import * 

def str2bool(v):
  return v.lower() in ('true', '1') 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--min_lr', default=1e-4, type=float, help='Min of learning rate')
parser.add_argument('--max_epoches', default=20, type=int, help='Max number of epoches')
parser.add_argument('--GPU', default=1, type=int, help='GPU core')
parser.add_argument('--use_GPU', default='true', type=str2bool, help='Use GPU or not')
parser.add_argument('--model', default='/home/tynguyen/cis680/logs/HW3/part1/mobilenet', type=str, help='Model path')
parser.add_argument('--data_path', default='/home/tynguyen/cis680/data/cifar10', type=str, help='Data path')
parser.add_argument('--resume', default='false', type=str2bool, help='resume from checkpoint')
parser.add_argument('--visual', default='false', type=str2bool, help='Display images')
parser.add_argument('--optim', default='adam', type=str, help='Type of optimizer', choices=['adam', 'sgd'])
args = parser.parse_args()
use_cuda = False 
if args.use_GPU:
  use_cuda = torch.cuda.is_available()
#use_cuda = False 

if use_cuda:
  torch.cuda.set_device(args.GPU) # set GPU that we want to use 
print('>> Current GPU', torch.cuda.current_device()) 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Prepare data 
print('===> Preparing data....') 

data_path = args.data_path 

# First, download the data 
#rawData_prepare = rawData_prepare(data_path)
#mean, std = rawData_prepare.download_tar(data_path)


param = {} 
#param['mean'] = (125.92819585, 123.48458521, 114.44250273)  
#param['std']  = ( 63.02008937,   62.1541447,  66.83732566)  

param['mean'] = (0.4914, 0.4822, 0.4465)
param['std'] =  (0.247, 0.243, 0.261)

param['data_path'] = os.path.join(data_path, 'imgs')  
param['label_path'] = os.path.join(data_path, 'train.txt') 

rescale = Rescale((32,32)) 
fliphorizontal = RandomFlipHorizontal() 
normalize = Normalize(param['mean'], param['std']) 
randomcrop = RandomCrop(32, 4) 
toTensor = ToTensor() 

composed = transforms.Compose([rescale,fliphorizontal, normalize,randomcrop , toTensor])
#composed = toTensor 
trainset = dataloader_obj(param, composed)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4) 


param['label_path'] = os.path.join(data_path, 'test.txt') 
testset = dataloader_obj(param, composed ) 
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Model 
if args.resume:
  # Load checkpoint 
  print('==> Resuming from checkpoint..') 
  assert os.path.exists(args.model), 'Error: no checkpoint directory found!'
  checkpoint =  torch.load(os.path.join(args.model, 'ckpt.t7')) 
  net = checkpoint['net'] 
  best_acc = checkpoint['acc'] 
  start_epoch = checkpoint['epoch'] 

else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    #net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # CIS 680 options 
    #net = ConvNet() # Part 1.1 
    net = MobileNet680() # Part 1.2  
if use_cuda:
  net.cuda() 
  #net.torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
  cudnn.benchmark = True 


# Loss function 
criterion = nn.CrossEntropyLoss() 

# Optimizer 
# Find learning rate decay factor 
lr_decay_f = (args.min_lr/args.lr)**(1./(args.max_epoches) )
if args.optim == 'sgd':
  optimizer  = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
else:
  optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=0.05)  
# Training
lr_array = []   

def adjust_lr(optimizer, lr_decay_factor):
  for param_group in optimizer.param_groups:
    cur_lr = param_group['lr']
    lr = cur_lr*lr_decay_factor 
    param_group['lr'] = lr
  print('===> cur_lr %.5f, updated lr %.5f, decay factor %.4f'%(cur_lr, lr, lr_decay_factor))
  return lr 
 
def train(epoch, max_iter=None, lr=0, visual=False):
  print('\nEpoch: %d' % epoch)
  net.train() 
  train_loss = 0 
  correct = 0 
  total = 0
  num_iter = 0  
  for batch_idx, sample_batched in enumerate(trainloader):
    print(batch_idx, sample_batched['image'].size(), sample_batched['label'].view(-1).size()) 
    if batch_idx == 0 and visual==True:
      plt.figure() 
      batch_display(sample_batched)
      plt.axis('off')
      plt.show() 
    inputs = sample_batched['image']
    targets = sample_batched['label']
    if use_cuda:
      inputs, targets = inputs.cuda(), targets.cuda() 
    optimizer.zero_grad() 
    inputs, targets = Variable(inputs) , Variable(targets.view(-1))
    outputs = net(inputs)
    loss  = criterion(outputs, targets)
    loss.backward() # Computer gradients 
    optimizer.step()  # Update network's parameters 

    train_loss += loss.data[0] 
    _, predicted = torch.max(outputs.data, 1) 
    total += targets.size(0) 
    correct += predicted.eq(targets.data).cpu().sum()
    
    epoch_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.4f'
          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, lr))
    num_iter+= 1 
    
    
    if max_iter and num_iter >= max_iter:
      break 

  # Update learning rate after each epoch 
  lr_array.append(lr) 
  lr = adjust_lr(optimizer, lr_decay_f) 
  return lr, epoch_time, 100.*correct/total  

def test(epoch, max_batches, visual=False):
  global best_acc 
  net.eval() 
  test_loss = 0 
  correct = 0 
  total = 0 
  num_batches = 0 
  for batch_idx, sample_batched in enumerate(testloader):
    print(batch_idx, sample_batched['image'].size(), sample_batched['label'].view(-1).size()) 
    if batch_idx == 0 and visual==True:
      plt.figure() 
      batch_display(sample_batched)
      plt.axis('off')
      plt.show() 
    inputs = sample_batched['image']
    targets = sample_batched['label']
    if use_cuda:
      inputs, targets = inputs.cuda(), targets.cuda() 
    optimizer.zero_grad() 
    inputs, targets = Variable(inputs), Variable(targets.view(-1))
    outputs = net(inputs)
    loss  = criterion(outputs, targets)
    
    test_loss += loss.data[0] 
    _, predicted = torch.max(outputs.data, 1) 
    total += targets.size(0) 
    correct += predicted.eq(targets.data).cpu().sum()

    progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    num_batches += 1 
    if num_batches >= max_batches:
      break 
  # Save checkpoint.
  acc = 100.*correct/total
  if acc > best_acc:
      print('Saving..')
      state = {
          'net': net, #net.module if use_cuda else net,
          'acc': acc,
          'epoch': epoch, 
      }
      if not os.path.exists(args.model):
          os.makedirs(args.model, exist_ok=True) # exist_ok: allows recursive (mkdir -p) 
      torch.save(state, os.path.join(args.model,'ckpt.t7') )
      best_acc = acc
  return  100.*correct/total 

lr = args.lr # Initial learning rate 
training_time = 0 
epoch_time = 0 
train_acc_array = [] 
test_acc_array = [] 
for epoch in range(start_epoch, start_epoch+args.max_epoches):
  lr, epoch_time, train_acc = train(epoch, 100, lr, args.visual)
  training_time += epoch_time 
  if epoch == args.max_epoches-1:
    max_batches = 100
  else: 
    max_batches = 5
  test_acc = test(epoch, max_batches)
  
  train_acc_array.append([epoch, training_time, train_acc])
  test_acc_array.append([epoch, training_time, test_acc]) 
  
  sys.stdout.write('\n=================================================================================\n')
  progress_bar(epoch, args.max_epoches, 'Current epoch: %d, tot_time: %.3f, epoch_time: %.3f******'%(epoch, training_time, epoch_time))  


# Save the results
train_acc_array = np.array(train_acc_array)  
test_acc_array = np.array(test_acc_array)  
np.savetxt(os.path.join(args.model, 'train_accuracy.txt'), train_acc_array) 
np.savetxt(os.path.join(args.model, 'test_accuracy.txt'), test_acc_array) 

# Plot results 
fig = plt.figure() 
plt.plot(train_acc_array[:, 2], color='b') 
plt.plot(test_acc_array[:, 2], color='r')
plt.xlabel('x 100 (Iterations)') 
plt.ylabel('Accuracy') 
plt.title('Training vs Testing Accuracy') 
plt.legend(['Train', 'Test'])
plt.savefig(os.path.join(args.model, 'accuracy.png'))   














