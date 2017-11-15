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
import pdb 
from models import * 
import time 
from utils import progress_bar, batch_display 
from data_loaders import * 
import utils 

def str2bool(v):
  return v.lower() in ('true', '1') 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--min_lr', default=1e-4, type=float, help='Min of learning rate')
parser.add_argument('--max_epoches', default=20, type=int, help='Max number of epoches')
parser.add_argument('--GPU', default=1, type=int, help='GPU core')
parser.add_argument('--use_GPU', default='true', type=str2bool, help='Use GPU or not')
parser.add_argument('--model', default='/home/tynguyen/cis680/logs/HW3/part2/2.1', type=str, help='Model path')
parser.add_argument('--data_path', default='/home/tynguyen/cis680/data/cifar10_transformed', type=str, help='Data path')
parser.add_argument('--resume', default='false', type=str2bool, help='resume from checkpoint')
parser.add_argument('--visual', default='false', type=str2bool, help='Display images')
parser.add_argument('--optim', default='adam', type=str, help='Type of optimizer', choices=['adam', 'sgd'])
parser.add_argument('--net', default='convnet', type=str, help='Type of nets', choices=['convnet', 'mobilenet', 'resnet', 'fasterrcnnnet'])
parser.add_argument('--loss_type', default='total', type=str, help='Type of loss functions', choices=['total','cls', 'reg'])
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

# Learning rates 
lr_array = []   

# Find anchors (x_a, y_a, w_a) N x 3 x 36 
mask_indices = utils.map_mask_2_img_coordinates()  # (2 x 36)
# TODO: set this value to a proper value
w_a = 40 # I think it should be 40 but TA said it is 32 
w_a_array = w_a*np.ones(36)
mask_indices = np.vstack([mask_indices, w_a_array])
 
anchors_tensor = torch.from_numpy(mask_indices).float()


def adjust_lr(optimizer, lr_decay_factor):
  for param_group in optimizer.param_groups:
    cur_lr = param_group['lr']
    lr = cur_lr*lr_decay_factor 
    param_group['lr'] = lr
  print('===> cur_lr %.5f, updated lr %.5f, decay factor %.4f'%(cur_lr, lr, lr_decay_factor))
  return lr 
 
def train(epoch, max_iter=None, lr=0, visual=False, is_train=True, best_acc=None):
  print('\nEpoch: %d' % epoch)
  if is_train:
    net.train() 
  else:
    net.eval()

  train_loss = 0 
  train_class_loss = 0 
  train_reg_loss = 0 

  #correct = 0 
  #total = 0
  num_iter = 0  
  dataloader_queue = trainloader if is_train else testloader

  for batch_idx, sample_batched in enumerate(dataloader_queue):
    batch_size = sample_batched['image'].size(0) 
    # print(batch_idx, sample_batched['image'].size(), sample_batched['label'].view(-1).size(), sample_batched['mask'].view(-1).size() ) 
    if batch_idx == 0 and visual==True:
      plt.figure() 
      batch_display(sample_batched)
      plt.axis('off')
      plt.show()
    # Input data  
    inputs = sample_batched['image']
    targets = sample_batched['label']
    masks = sample_batched['mask']
    boxes = sample_batched['box'].view(batch_size, 3).float()
    anchors = anchors_tensor
    if use_cuda:
      inputs, targets, masks, boxes, anchors = inputs.cuda(), targets.cuda(), masks.cuda(), boxes.cuda(), anchors_tensor.cuda()  
    # Reshape masks to (N x 36) 
    masks = masks.view(batch_size,-1) 
    # Create a mask to ignore all 2-elements (white in the mask)
    neq_two_filter = masks.le(1).float()  # N x 36 
    one_filter = (masks == 1).float()
    zero_filter = (masks == 0).float()
      
    
    # Loss function for object vs non object  
    isobject_criterion = nn.BCEWithLogitsLoss(neq_two_filter) 
  
    # Predict output 
    if is_train:
      optimizer.zero_grad() 
    inputs, targets, masks, boxes = Variable(inputs) , Variable(targets.view(-1)), Variable(masks), Variable(boxes)
    anchors = Variable(anchors)
    one_filter, zero_filter = Variable(one_filter), Variable(zero_filter)
    
    outputs = net(inputs) 
    isobject_outputs =  outputs['cls']['out'].view(batch_size, -1) # output: (N x 36)
    reg_outputs = outputs['reg']['out'] # N x 3 x 36

    # pdb.set_trace()
  
    reg_loss = utils.get_reg_loss(reg_outputs, boxes, one_filter, anchors) # Sum over minibatch 
    print('==> Net %s | Type Loss: %s | Reg loss %.4f over %d total boxes| '%(args.net, args.loss_type, reg_loss.data[0], one_filter.data.sum()))
    
    
    # Get accuracy 
    #max_outputs, _ = torch.max(isobject_outputs, 1, keepdim=True) 
    #center_predict = isobject_outputs.eq(max_outputs)
    #total += batch_size

    #correct += torch.masked_select(masks.view(, center_predict).eq(1).float().cpu().sum().data.numpy()[0]  
    
    # Get accuracy 
    pos_thresh = 0.5 
    use_sigmoid = True 
    if use_sigmoid:
      sigmoided_isobject_outputs = torch.sigmoid(isobject_outputs)
    else:
      sigmoided_isobject_outputs = isobject_outputs
    pos_pred = (sigmoided_isobject_outputs >= pos_thresh).float()
    neg_pred = (sigmoided_isobject_outputs < pos_thresh).float() 
    correct  = (pos_pred * one_filter).sum().cpu().data.numpy()[0] + (neg_pred * zero_filter).sum().cpu().data.numpy()[0]
    total    =  one_filter.sum().cpu().data.numpy()[0] + zero_filter.sum().cpu().data.numpy()[0]
    #print('Correct:', correct, 'total:', total)  
    class_loss  = isobject_criterion(isobject_outputs.view(-1), masks.view(-1)) # evarage over minibatch 

    # test only reg_loss
    use_loss_type  = args.loss_type  
    if use_loss_type == 'cls':
      total_loss = class_loss
    elif use_loss_type == 'reg':
      total_loss = reg_loss 
    else:
      total_loss = reg_loss + class_loss   # multiply 10 to make regression run faster
    
    if is_train:
      total_loss.backward() # Computer gradients 
      optimizer.step()  # Update network's parameters 

    train_class_loss += class_loss.data[0] 
    train_reg_loss += reg_loss.data[0] 
    train_loss += total_loss.data[0]
    #epoch_time = progress_bar(batch_idx, len(trainloader), 'Is train: %d  | Total Loss: %.3f | Class Loss: %.4f |Reg Loss: %.4f |Acc: %.3f%% | (%d/%d) | lr: %.6f'
        #  % (int(is_train), train_loss/(batch_idx+1), train_class_loss/(batch_idx+1), train_reg_loss/(batch_idx+1), 100.*correct/total, correct, total,  lr))
    
    epoch_time = progress_bar(batch_idx, len(trainloader), 'Is train: %d  | Total Loss: %.3f | Class Loss: %.4f |Reg Loss: %.4f |Acc: %.3f%% | (%d/%d) | lr: %.6f'
          % (int(is_train), total_loss.data[0], class_loss.data[0], reg_loss.data[0], 100.*correct/total, correct, total,  lr))
    
    num_iter+= 1 
    
    if max_iter and num_iter >= max_iter:
      break   

  # Save checkpoint.
  if not is_train:
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

  # Update learning rate after each epoch 
  lr_array.append(lr) 
  lr = adjust_lr(optimizer, lr_decay_f) 
  return lr, epoch_time, 100.*correct/total, train_loss/(batch_idx+1), train_class_loss/(batch_idx+1), train_reg_loss/(batch_idx+1), best_acc

 
def run():
  lr = args.lr # Initial learning rate 
  training_time = 0 
  epoch_time = 0 

  # Two big arrays to log results 
  train_acc_array = [] 
  test_acc_array = [] 
  best_acc = 0 

  for epoch in range(start_epoch, start_epoch+args.max_epoches):
    lr, epoch_time, train_acc, train_loss, train_class_loss, train_reg_loss, _ = train(epoch, 100, lr, args.visual)
    training_time += epoch_time 
    if epoch == args.max_epoches-1:
      max_batches = 100
    else: 
      max_batches = 2 
    

    _, _, test_acc, test_loss, test_class_loss, test_reg_loss, best_acc = train(epoch, max_batches, is_train=False, best_acc=best_acc)
    
    train_acc_array.append([epoch, training_time, train_acc, train_loss, train_class_loss, train_reg_loss])
    test_acc_array.append([epoch, training_time, test_acc, test_loss, test_class_loss, test_reg_loss]) 
    
    sys.stdout.write('\n=================================================================================\n')
    progress_bar(epoch, args.max_epoches, 'Current epoch: %d, tot_time: %.3f, epoch_time: %.3f******'%(epoch, training_time, epoch_time))  


  # Save the results
  train_acc_array = np.array(train_acc_array)  
  np.savetxt(os.path.join(args.model, 'train_accuracy.txt'), train_acc_array) 


  test_acc_array = np.array(test_acc_array)  
  np.savetxt(os.path.join(args.model, 'test_accuracy.txt'), test_acc_array) 

  # Plot results 
  plt.figure(figsize=(12,12))
  plt.subplot(221)
  plt.plot(train_acc_array[:, 2], color='b') 
  plt.plot(test_acc_array[:, 2], color='r')
  plt.xlabel('x 100 (Iterations)') 
  plt.ylabel('Accuracy') 
  plt.title('Training vs Testing Accuracy') 
  plt.legend(['Train', 'Test'])
  plt.savefig(os.path.join(args.model, 'accuracy.png'))   

  plt.subplot(222)
  plt.plot(train_acc_array[:, 3], color='b') 
  plt.plot(test_acc_array[:, 3], color='r')
  plt.xlabel('x 100 (Iterations)') 
  plt.ylabel('Total Loss') 
  plt.title('Training vs Testing Total Loss') 
  plt.legend(['Train', 'Test'])
  plt.savefig(os.path.join(args.model, 'total_loss.png'))   


  plt.subplot(223)
  plt.plot(train_acc_array[:, 4], color='b') 
  plt.plot(test_acc_array[:, 4], color='r')
  plt.xlabel('x 100 (Iterations)') 
  plt.ylabel('Classification Loss') 
  plt.title('Training vs Testing Classification Loss') 
  plt.legend(['Train', 'Test'])
  plt.savefig(os.path.join(args.model, 'class_loss.png'))   

  plt.subplot(224)
  plt.plot(train_acc_array[:, 5], color='b') 
  plt.plot(test_acc_array[:, 5], color='r')
  plt.xlabel('x 100 (Iterations)') 
  plt.ylabel('Regression Loss') 
  plt.title('Training vs Testing Regression Loss') 
  plt.legend(['Train', 'Test'])
  plt.savefig(os.path.join(args.model, 'Regression_loss.png'))   





# Run the experiments 
run()
