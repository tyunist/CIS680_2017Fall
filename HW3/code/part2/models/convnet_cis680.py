# Convnet used in part1, cis 680 homework 3 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch.autograd import Variable 

cfg = [(5, 5, 32), # Conv1 
       ('M',2,2,0),  # MaxPool1
       (5, 5, 64), # Conv2  
       ('M',2,2,0),  # MaxPool2
       (5, 5, 128),# Conv3 
       ('M',2,2,0),  # MaxPool3
       (5, 5, 256),# Conv4 
       ('M',2,2,0),  # MaxPool4
       (3, 3, 512),# Conv5 
       ('M',2,2,0),  # MaxPool5
      ]   
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__() 
    self.features = self._make_layers(cfg) 
    self.classifier = nn.Linear(512, 10)  
  
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
      if len(x) == 4: # MaxPooling 
        layers += [nn.MaxPool2d(kernel_size=x[1], stride=x[2], padding=x[3])]
      else:           # Conv followed by Batchnorm and ReLue 
        layers += [nn.Conv2d(in_channels, out_channels=x[2], kernel_size=(x[0], x[1]), stride=1, padding=(int(x[0]/2), int(x[1]/2)), bias=False ), 
                 nn.BatchNorm2d(x[2]), 
                 nn.ReLU(inplace=True)]
        in_channels = x[2]
    return nn.Sequential(*layers)


def test():
  net = ConvNet()
  x = torch.randn(1, 3, 32, 32) 
  y = net(Variable(x)) 
  print(y.size())

if __name__ == "__main__":
  test() 
