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
parser.add_argument('--model', default='/home/tynguyen/cis680/logs/HW3/part1/', type=str, help='Model path')
parser.add_argument('--data_path', default='/home/tynguyen/cis680/data/cifar10_transformed', type=str, help='Data path')
parser.add_argument('--resume', default='false', type=str2bool, help='resume from checkpoint')
parser.add_argument('--visual', default='false', type=str2bool, help='Display images')
parser.add_argument('--optim', default='adam', type=str, help='Type of optimizer', choices=['adam', 'sgd'])
parser.add_argument('--net', default='convnet', type=str, help='Type of nets', choices=['convnet', 'mobilenet', 'resnet', 'fasterrcnnnet'])
args = parser.parse_args()
use_cuda = False 
if args.use_GPU:
  use_cuda = torch.cuda.is_available()
#use_cuda = False 


# Logfile
args.model = os.path.join(args.model, args.net) 

if use_cuda:
  torch.cuda.set_device(args.GPU) # set GPU that we want to use 
print('>> Current GPU', torch.cuda.current_device()) 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Prepare data 
print('===> Preparing data....') 

data_path = args.data_path 



param = {} 


param['data_path'] = os.path.join(data_path, 'imgs')  
param['label_path'] = os.path.join(data_path, 'train.txt') 
param['mask_path'] = os.path.join(data_path, 'masks') 

toTensor = Cifar10_transformed_ToTensor() 

composed = transforms.Compose([toTensor])
trainset = cifar10_transformed_loader_obj(param, composed)
BATCH_SIZE = 100
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1) 


param['label_path'] = os.path.join(data_path, 'test.txt') 
testset = cifar10_transformed_loader_obj(param, composed)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

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
    if args.net == 'convnet':
      net = ConvNet() # Part 1.1 
    elif args.net == 'mobilenet':
      net = MobileNet680() # Part 1.2  
    elif args.net == 'resnet':
      net = ResNet680() # Part 1.3  
    elif args.net == 'basenet':
      net = BaseNet680()
    elif args.net == 'fasterrcnnnet':
      net = Faster_RCNN_net680() 

if use_cuda:
  net.cuda() 
  #net.torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
  cudnn.benchmark = True 



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
    batch_size = sample_batched['image'].size(0) 
    print(batch_idx, sample_batched['image'].size(), sample_batched['label'].view(-1).size(), sample_batched['mask'].view(-1).size() ) 
    if batch_idx == 0 and visual==True:
      plt.figure() 
      batch_display(sample_batched)
      plt.axis('off')
      plt.show()
    # Input data  
    inputs = sample_batched['image']
    targets = sample_batched['label']
    masks = sample_batched['mask']
    # Reshape masks to (n x 36, ) 
    masks = masks.view(-1) 
    # Create a mask to ignore all 2-elements (white in the mask)
    value_filter = masks.le(1).float()  
    
    # Loss function for object vs non object  
    isobject_criterion = nn.BCEWithLogitsLoss(value_filter) 
  
    if use_cuda:
      inputs, targets, masks = inputs.cuda(), targets.cuda(), mask.cuda() 
    # Predict output 
    optimizer.zero_grad() 
    inputs, targets, masks = Variable(inputs) , Variable(targets.view(-1)), Variable(masks)
    isobject_outputs =  net(inputs)['out'].view(-1) # output: (N x 36, )
    
    # Get accuracy 
    max_outputs, _ = torch.max(isobject_outputs.view(batch_size, -1), 1, keepdim=True) 
    center_predict = isobject_outputs.view(batch_size, -1).eq(max_outputs)
    total += batch_size

    correct += torch.masked_select(masks.view(batch_size, -1), center_predict).eq(1).float().sum().data.numpy()[0]  
    loss  = isobject_criterion(isobject_outputs, masks)
    loss.backward() # Computer gradients 
    optimizer.step()  # Update network's parameters 

    train_loss += loss.data[0] 
    epoch_time = progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.4f'
          % (train_loss/(batch_idx+1), 100.*correct/total, correct, total,  lr))
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
  num_iter = 0  
  for batch_idx, sample_batched in enumerate(testloader):
    batch_size = sample_batched['image'].size(0) 
    print(batch_idx, sample_batched['image'].size(), sample_batched['label'].view(-1).size()) 
    if batch_idx == 0 and visual==True:
      plt.figure() 
      batch_display(sample_batched)
      plt.axis('off')
      plt.show() 
    inputs = sample_batched['image']
    targets = sample_batched['label']
    masks = sample_batched['mask']
    # Reshape masks to (n x 36, ) 
    masks = masks.view(-1) 
    # Create a mask to ignore all 2-elements (white in the mask)
    value_filter = masks.le(1).float() 
    one_filter = masks.eq(1).float() 
    
    if use_cuda:
      inputs, targets, masks = inputs.cuda(), targets.cuda(), mask.cuda() 
    optimizer.zero_grad() 
    inputs, targets, masks = Variable(inputs) , Variable(targets.view(-1)), Variable(masks)
    isobject_outputs = net(inputs)['out'].view(-1)  
    
    max_outputs, _ = torch.max(isobject_outputs.view(batch_size, -1), 1, keepdim=True) 
    center_predict = isobject_outputs.view(batch_size, -1).eq(max_outputs)
    
    total += float(batch_size)
    
    correct += torch.masked_select(masks.view(batch_size, -1), center_predict).eq(1).float().sum().data.numpy()[0]  
    
    # Loss criterion 
    isobject_criterion = nn.BCEWithLogitsLoss(value_filter) 
    # Reshape output 
    loss  = isobject_criterion(isobject_outputs, masks)
    
    test_loss += loss.data[0] 
    
    epoch_time = progress_bar(batch_idx, len(testloader), 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d) | lr: %.4f'
          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total,  lr))
    num_iter+= 1 
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














