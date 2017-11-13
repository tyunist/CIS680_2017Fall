import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable 
import torch.nn.init as init
import time  
basenet_cfg = [(5, 5, 32), 
       ('M', 2, 2, 0), 
       (5, 5, 64), 
       ('M', 2, 2, 0),
       (5, 5, 128), 
       ('M', 2, 2, 0), 
       (3, 3, 256)]

class_proposal_cfg = [(3, 3, 256), 
                      (1, 1, 1)]

box_regression_cfg = [(1, 1, 3)]


def init_weight_params(net, method_name='xavier'):
  for m in net.modules():
    if isinstance(m, nn.Conv2d):
      if method_name == 'xavier':
        init.xavier_uniform(m.weight.data)
        if m.bias is not None:  
          # if bias has only one dimension, just initialize using normal distribution
          if m.bias.data.ndimension() < 2:
            m.bias.data.normal_(0,1) 
          else: 
            init.xavier_uniform(m.bias.data)
    if isinstance(m, nn.Linear):
      if method_name == 'xavier':
        init.xavier_uniform(nn.weight.data)

class BoxRegressionNet680(nn.Module):
  """Regress the box. 
     Input: features obtained from intermediate layer (256 x d) 
     Output: 4k coordinates"""
  def __init__(self, input_channels=3):
    super(BoxRegressionNet680, self).__init__() 
    self.cfg = box_regression_cfg[0] 
    self.conv = nn.Conv2d(input_channels, self.cfg[2], kernel_size=(self.cfg[0], self.cfg[1]), stride=1, padding=0, bias=True) 
    # Initialize bias for convolutional layer 
    self.conv.bias.data = torch.FloatTensor([24,24,32])
  def forward(self, x):
    self.out_conv = self.conv(x)
    self.out = self.out_conv.view(self.out_conv.size(0), self.cfg[2], -1) # N x 3 x 36 
    return {'out': self.out}

class ClassProposalNet680(nn.Module):
  """Input is the features obtained from a CNN (without Fully-connected layers).
     Output: N x (6x6) for given input images and network in cis680 HW3 part 2"""
  def __init__(self, input_channels=3):
    super(ClassProposalNet680, self).__init__() 
    c1_cfg = class_proposal_cfg[0]
    c1_padding = (int(c1_cfg[0]/2), int(c1_cfg[1]/2))
    c1_kernel = (c1_cfg[0], c1_cfg[1]) 
    self.conv1 = nn.Conv2d(input_channels,c1_cfg[2], kernel_size=c1_kernel, stride=1, padding=c1_padding, bias=False)
    input_channels = c1_cfg[2] 
    self.bn1 = nn.BatchNorm2d(c1_cfg[2]) 
    
    c2_cfg = class_proposal_cfg[1]
    c2_padding = (int(c2_cfg[0]/2), int(c2_cfg[1]/2))
    c2_kernel = (c2_cfg[0], c2_cfg[1]) 
    self.conv2 = nn.Conv2d(input_channels,c2_cfg[2], kernel_size=c2_kernel, stride=1, padding=c2_padding, bias=True)
  
  def forward(self,x):
    self.out_conv1 = F.relu(self.bn1(self.conv1(x)))
    self.out_conv2 = torch.squeeze(self.conv2(self.out_conv1))
    # Reshape to n x 36  
    self.out = self.out_conv2.view(self.out_conv2.size(0),-1)   
    return {'intermediate':self.out_conv1,
            'conv2':self.out_conv2, 
            'out': self.out_conv2} 
    

class BaseNet680(nn.Module):
  def __init__(self, input_channels=3):
    super(BaseNet680,self).__init__() 
    # Conv1, bn1 and pool1 
    c1_cfg = basenet_cfg[0]
    c1_padding = (int(c1_cfg[0]/2), int(c1_cfg[1]/2))
    c1_kernel = (c1_cfg[0], c1_cfg[1]) 
    self.conv1 = nn.Conv2d(input_channels,c1_cfg[2],kernel_size=c1_kernel, stride=1, padding=c1_padding, bias=False) 
    input_channels = c1_cfg[2] 
    
    self.bn1 = nn.BatchNorm2d(input_channels)
    
    p1_cfg = basenet_cfg[1]
    self.pool1 = nn.MaxPool2d(kernel_size=p1_cfg[1], stride=p1_cfg[2], padding=p1_cfg[3])    
    
    # Conv2, bn2 and pool2 
    c2_cfg = basenet_cfg[2]
    c2_padding = (int(c2_cfg[0]/2), int(c2_cfg[1]/2)) 
    c2_kernel = (c2_cfg[0], c2_cfg[1]) 
    self.conv2 = nn.Conv2d(input_channels,c2_cfg[2],kernel_size=c2_kernel, stride=1, padding=c2_padding, bias=False) 
    input_channels = c2_cfg[2] 
    
    self.bn2 = nn.BatchNorm2d(input_channels)
    
    p2_cfg = basenet_cfg[3]
    self.pool2 = nn.MaxPool2d(kernel_size=p2_cfg[1], stride=p2_cfg[2], padding=p2_cfg[3])    
    
    # Conv3, bn3 and pool3 
    c3_cfg = basenet_cfg[4]
    c3_padding = (int(c3_cfg[0]/2), int(c3_cfg[1]/2))
    c3_kernel = (c3_cfg[0], c3_cfg[1]) 
    self.conv3 = nn.Conv2d(input_channels,c3_cfg[2],kernel_size=c3_kernel, stride=1, padding=c3_padding, bias=False) 
    input_channels = c3_cfg[2] 
    
    self.bn3 = nn.BatchNorm2d(input_channels)
    
    p3_cfg = basenet_cfg[5]
    self.pool3 = nn.MaxPool2d(kernel_size=p3_cfg[1], stride=p3_cfg[2], padding=p3_cfg[3])
    
    # Conv4 and bn4 
    c4_cfg = basenet_cfg[6] 
    c4_padding = (int(c4_cfg[0]/2), int(c4_cfg[1]/2)) 
    c4_kernel = (c4_cfg[0], c4_cfg[1]) 
    self.conv4 = nn.Conv2d(input_channels,c4_cfg[2],kernel_size=c4_kernel, stride=1, padding=c4_padding, bias=False)
    
    self.bn4 = nn.BatchNorm2d(c4_cfg[2])

  def forward(self, x):
    self.out_conv1 = F.relu(self.bn1(self.conv1(x)))
    self.out_pool1 = self.pool1(self.out_conv1)
    self.out_conv2 = F.relu(self.bn2(self.conv2(self.out_pool1))) 
    self.out_pool2 = self.pool2(self.out_conv2)
    self.out_conv3 = F.relu(self.bn3(self.conv3(self.out_pool2))) 
    self.out_pool3 = self.pool3(self.out_conv3)
    self.out_conv4 = F.relu(self.bn4(self.conv4(self.out_pool3))) 
    return {'conv1': self.out_conv1, 
            'conv2': self.out_conv2, 
            'conv3': self.out_conv3, 
            'out': self.out_conv4  
           }
class Faster_RCNN_net680(nn.Module):
  def __init__(self, in_channels=3):
    super(Faster_RCNN_net680, self).__init__() 
    self.basenet = BaseNet680(in_channels)
    in_channels = basenet_cfg[6][2] 
    self.classnet = ClassProposalNet680(in_channels)
    in_channels = class_proposal_cfg[0][2]
    self.regressionnet = BoxRegressionNet680(in_channels)
    # Initialize weights for classnet and basenet 
    self.basenet.apply(init_weight_params)  
    self.classnet.apply(init_weight_params)
    #self.regressionnet.apply(init_box_regression_params) 
    

  def forward(self, x):
    self.out_basenet = self.basenet(x)['out']
    self.out_classnet = self.classnet(self.out_basenet)
    self.out_regressionnet = self.regressionnet(self.out_classnet['intermediate'])     
    return {'cls':self.out_classnet, 
            'reg':self.out_regressionnet} 

def test_separate():
  basenet = BaseNet680()
  x = torch.randn(2,3,48,48)
  out = basenet(Variable(x)) 
  print('conv1 size:', out['conv1'].size())
  print('conv2 size:', out['conv2'].size())
  print('conv3 size:', out['conv3'].size())
  print('conv4 size:', out['out'].size())
  
  out_basenet_channels = out['conv4'].size(1) 
  classnet = ClassProposalNet680(out_basenet_channels)
 
  out = out['conv4'] 
  out = classnet(out) 
  print('conv1 size:', out['intermediate'].size())
  print('conv2 size:', out['conv2'].size())
  print('out class size:', out['out'].size())

def test_faster_rcnn_net():
  fasterrcnnnet = Faster_RCNN_net680()
  x = torch.randn(2,3,48,48)
  out = fasterrcnnnet(Variable(x))
  
  regressionnet = fasterrcnnnet.regressionnet
  for m in regressionnet.modules():
    if isinstance(m, nn.Conv2d):
      print('Bias:', m.bias.data) 
  print('class proposal out size:', out['cls']['out'].size())
  print('intermediate size:', out['cls']['intermediate'].size())
  print('Reg out size:', out['reg']['out'].size())
  
if __name__=="__main__":
  test_faster_rcnn_net() 
  #test() 
