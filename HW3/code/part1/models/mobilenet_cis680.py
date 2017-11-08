# Convnet used in part1, cis 680 homework 3 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch.autograd import Variable 

cfg = [('B', 5, 5, 32), # Conv1 
       ('M',2,2,0),  # MaxPool1
       ('B', 5, 5, 64), # Conv2  
       ('M',2,2,0),  # MaxPool2
       ('B', 5, 5, 128),# Conv3 
       ('M',2,2,0),  # MaxPool3
       ('B', 5, 5, 256),# Conv4 
       ('M',2,2,0),  # MaxPool4
       (3, 3, 512),# Conv5 
       ('M',2,2,0),  # MaxPool5
      ]   
class depth_block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel):
    super(depth_block, self).__init__()
    layers = [] 
    layers += [nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=(kernel[0], kernel[1]), stride=1, padding=(int(kernel[0]/2), int(kernel[1]/2)), groups=in_channels, bias=False), 
                  nn.BatchNorm2d(in_channels), 
                  nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1, padding=0, groups=1, bias=False), 
                  nn.BatchNorm2d(out_channels), 
                  nn.ReLU(inplace=True)]
    self.features =  nn.Sequential(*layers) 
  def forward(self, x):
    out = self.features(x)  
    return out 


class MobileNet680(nn.Module):
  def __init__(self):
    super(MobileNet680, self).__init__() 
    self.features = self._make_layers(cfg) 
    self.classifier = None 
  
  def forward(self, x):
    out = self.features(x) 
    print('===> Conv Out size:', out.size())
    out = out.view(out.size(0), -1) 
    if self.classifier:
      out = self.classifier(out)
    return out 
  
  def _make_layers(self, cfg):
    layers = [] 
    in_channels = 3 
    for x in cfg:
      if len(x) == 4: 
        if x[0] == 'M': # maxpooling 
          layers += [nn.MaxPool2d(kernel_size=x[1], stride=x[2], padding=x[3])]
        elif x[0] == 'B':
          layers += [depth_block(in_channels, x[3], (x[1], x[2]))]
          in_channels = x[3]
      else:           # Conv followed by Batchnorm and ReLue 
        layers += [nn.Conv2d(in_channels, out_channels=x[2], kernel_size=(x[0], x[1]), stride=1, padding=(int(x[0]/2), int(x[1]/2)), bias=False), 
                  nn.BatchNorm2d(x[2]),
                  nn.ReLU(inplace=True)] 
        in_channels = x[2]
    return nn.Sequential(*layers)


def test():
  net = MobileNet680()
  x = torch.randn(1, 3, 32, 32) 
  y = net(Variable(x)) 
  print(y.size())

if __name__ == "__main__":
  test() 
