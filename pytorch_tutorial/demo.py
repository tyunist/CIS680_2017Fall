import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import torch,numpy as np
import pickle 

param = {}
param['data_path'] = #specify data path
param['label_path'] = #specify data label path
param['mean_val'] = #specify the mean value 
param['mean_std'] = #specify the standard variant
param['total_epoch'] = #define the epoch number here

# define the model that contains layers and functions
class cifar_10_model(nn.Module):
  def __init__(self):
    super(cifar_10_model, self).__init__()
    # pre-define layer here 
    self.conv1 = nn.Conv2d(3, 32, 5, stride = 1, padding = 2)
    self.conv2 = nn.Conv2d(32, 32, 5, stride = 1, padding = 2)
    self.conv3 = nn.Conv2d(32, 64, 5, stride = 1, padding = 2)
    self.fc1   = nn.Linear(8*8*64, 64)
    self.fc2   = nn.Linear(64, 10)
    self.bn1   = nn.BatchNorm2d(32)
    self.bn2   = nn.BatchNorm2d(32)
    self.bn3   = nn.BatchNorm2d(64)
    self.bn4   = nn.BatchNorm2d(64)

  def forward(self, x):
    # do the forward computation
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = F.avg_pool2d(x, 2)
    x = self.conv2(x)
    x = F.relu(self.bn2(x))
    x = F.avg_pool2d(x, 2)
    x = self.conv3(x)
    x = F.relu(self.bn3(x))
    x = F.avg_pool2d(x, 2)
    # reshape x from 4D to 2D, before reshape, x.size() = N*C*H*W, after reshape, x.size() = N*D
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    x = F.relu(self.bn4(x))
    x = self.fc2(x)
    return x

class dataloader_obj(data.Dataset):
  def __init__(self, param):
    f_data = open(param['data_path'], 'rb')
    self.data = pickle.load(f_data)
    #reshape to N*C*H*W
    self.data = self.data.reshape(self.data.shape[0],3,32,32)
    f_data.close()
    f_label = open(param['data_label'], 'rb')
    self.label = pickle.load(f_label)
    f_label.close()
    self.mean_val = param['mean_val']
    self.std = param['std']

  def __getitem__(self, index):
    #every time the data loader is called, it will input a index, 
    #the getitem function will return the image based on the index
    #the maximum index number is defined in __len__ method below
    #for each calling, you could do the image preprocessing, flipping or cropping
    img = self.data[index,:,:,:][np.newaxis,...]
    # use broadcasting to vectorizely normalize image
    img = (img - self.mean_val.reshape(1,3,1,1))/(self.std.reshape(1,3,1,1))
    label = self.label[index]
    # convert numpy array to torch tensor variable
    img = Variable(torch.from_numpy(img.astype(np.float32)))
    label = Variable(torch.from_numpy(label)).type(torch.LongTensor)
    return img, label

  def __len__(self):
    #this function define the upper bound of input index
    #it's usually set to the data image number
    return self.data.shape[0]

# initialized your defined model
model = cifar_10_model()
# initialized your define dataloader-obj
dataloader_obj = dataloader_obj(param)
# use torch powerful parallel dataloader, 
trainloader = torch.utils.data.DataLoader(dataloader_obj, batch_size=100, shuffle=True, num_workers=2)
# define optimizer to update the model
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
# define loss function
loss_fn = nn.CrossEntropyLoss()

#set model type to train
model.train()
for epoch_index in range(param['total_epoch']):
  for batch_idx, (img_data, img_label) in enumerate(trainloader):
    #feed input image to model and do the forward computation and get the result
    pred = model.forward(img_data)
    #compute the loss 
    loss = loss_fn(pred, img_label) 
    #remember to use optimizer.zero_grad() every time before compute gradient.
    #since, the grad will accumulate and will not be automatically cleaned
    #so if you don't clean gradient by .zero_grad(), previous step's gradient will 
    #be added to this step
    optimizer.zero_grad()
    #compute grad for all parameter related to the loss
    loss.backward()
    #update the parameter based on the grad
    optimizer.step()


#############################
# some additional operations# 
#############################

#######################################
#convert between variable and tensor
#create a torch tensor from numpy array
a = torch.from_numpy(np.arange(10))
# convert tensor to variable
b = Variable(a)
# convert Variable to tensor
c = b.data
# convert torch Variable to numpy array
b_numpy = b.data.numpy()
# convert torch tensor to numpy array
a_numpy = a.numpy()
#######################################


##################################################
# extract the weight or the gradient of parameter
conv1_weight = model.conv1.weight.data.numpy()
conv1_grad = model.conv1.weight.grad.data.numpy()
##################################################


#################################################
# Fuse multi model
pred1 = model1.forward(data1)
pred2 = model2.forward(data2)
loss1 = loss_fn(pred1,label1)
loss2 = loss_fn(pred2,label2)
loss = loss1 + loss2
loss.backward()
################################################


#################################################################################################
# use torch.nn and torch.nn.functional
input_dim = 3
output_dim = 64
kernele_size = 5
stride = 1
padding = 1

#pre-define nn.conv2d since it will automatically create a initialized parameter and store it inside
conv_nn = torch.nn.Conv2d(input_dim, output_dim, kernele_size, stride = stride, padding = padding)
#do forward computation use 
output1 = conv_nn(data)

#pre-define weight
weight = np.random.randn(output_dim, input_dim, kernel_size, kernel_size) * sqrt(2.0/input_dim)
weight = torch.Parameter(torch.from_numpy(weight))
#pre-define bias
bias = np.random.randn(output_dim) * sqrt(2.0)
bias= torch.Parameter(torch.from_numpy(bias)).view(1,output_dim,1,1)
# Don't need to pre-define conv2D function as nn.Conv2d, just put in the data and weight
output2 = torch.nn.functional.conv2d(data, weight, stride = stride, padding = padding) + bias
##################################################################################################


#####################################
#different way to pass parameter to optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD([param_1,param_2], lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD([
            {'params': model.conv1.parameters()}
            {'params': model.conv2.parameters(),'lr': 1e-3}
            {'params': model.conv3.parameters(),'lr': 1e-4}
                              ]
              , lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD([
            {'params': model1.parameters()}
            {'params': model2.parameters(),'lr': 1e-3}
            {'params': model2.parameters(),'lr': 1e-4}
                              ]
              , lr=0.01, momentum=0.9, weight_decay=5e-4)
##################################### 
