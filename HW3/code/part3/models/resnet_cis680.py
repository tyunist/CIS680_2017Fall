# Resnet used in part1, cis 680 homework 3 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch.autograd import Variable 

cfg = [('R', 5, 5, 32), # Conv1 
       ('M',2,2,0),  # MaxPool1
       ('R', 5, 5, 64), # Conv2  
       ('M',2,2,0),  # MaxPool2
       ('R', 5, 5, 128),# Conv3 
       ('M',2,2,0),  # MaxPool3
       ('R', 5, 5, 256),# Conv4 
       ('M',2,2,0),  # MaxPool4
       (3, 3, 512),# Conv5 
       ('M',2,2,0),  # MaxPool5
      ]   

class res_block_3(nn.Module):
  """Residual block with kernel of size 3x3"""
  def __init__(self, in_channels, planes):
    super(res_block_3,self).__init__()
    self.brach_3 = nn.Sequential(nn.Conv2d(in_channels, planes, kernel_size=3, stride=1, padding=1, bias=False), 
                    nn.BatchNorm2d(planes), 
                    nn.ReLU(inplace=True),
                    nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1,bias=False),  
                    nn.BatchNorm2d(planes)      
                ) 
    self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, planes, kernel_size=1, stride=1, padding=0, bias=False), 
                    nn.BatchNorm2d(planes)  
                   )
  def forward(self,x):
    main = self.brach_3(x) 
    shortcut = self.shortcut(x) 
    out = main + shortcut 
    out = F.relu(out)
    return out 

class ResNet680(nn.Module):
  def __init__(self):
    super(ResNet680, self).__init__() 
    self.features = self._make_layers(cfg) 
    self.classifier = nn.Linear(512,10) 
  
  def forward(self, x):
    out = self.features(x) 
    #out = torch.squeeze(out)
    out = out.view(out.size(0), -1)
    print('===> Conv Out size:', out.size())
    #out = F.dropout(out, training=self.training) 
    if self.classifier:
      out = self.classifier(out)
    return out 
  
  def _make_layers(self, cfg):
    layers = [] 
    in_channels = 3 
    for x in cfg:
      if len(x) == 4: 
        if x[0] == 'M': # MaxPooling 
          layers += [nn.MaxPool2d(kernel_size=x[1], stride=x[2], padding=x[3])]
        elif x[0] == 'R':
          layers += [res_block_3(in_channels, x[3]) ]
          in_channels = x[3] 
      else:           # Conv followed by Batchnorm and ReLu 
        layers += [nn.Conv2d(in_channels, out_channels=x[2], kernel_size=(x[0], x[1]), stride=1, padding=(int(x[0]/2), int(x[1]/2)), bias=False ), 
                 nn.BatchNorm2d(x[2]), 
                 nn.ReLU(inplace=True)]
        in_channels = x[2]
    return nn.Sequential(*layers)


def test():
  net = ResNet680()
  x = torch.randn(1, 3, 32, 32) 
  y = net(Variable(x)) 
  print(y.size())

if __name__ == "__main__":
  test() 
