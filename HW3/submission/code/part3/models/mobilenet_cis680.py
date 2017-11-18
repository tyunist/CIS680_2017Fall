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

def get_rep_field(n_in, j_in, r_in, start_in, kernel_size, stride, padding):
  n_out = (n_in + 2*padding - kernel_size)/stride + 1 
  j_out = j_in * stride 
  r_out = r_in + (kernel_size - 1)*j_in 
  start_out = start_in + ((kernel_size-1)/2 - padding)*j_in 
  return n_out, j_out, r_out, start_out 

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
    self.classifier = nn.Linear(512,10) 
  
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
    num_conv = 1
    num_multi = 0  
    n_in, j_in, r_in, start_in = 32, 1, 1, 0.5 
    for x in cfg:
      if len(x) == 4: 
        if x[0] == 'M': # maxpooling 
          layers += [nn.MaxPool2d(kernel_size=x[1], stride=x[2], padding=x[3])]
        elif x[0] == 'B':
          layers += [depth_block(in_channels, x[3], (x[1], x[2]))]

          n_out, j_out, r_out, start_out = get_rep_field(n_in, j_in, r_in, start_in, kernel_size=x[1], stride=1, padding=int(x[1]/2)) 
   
          mutltis =  n_in*n_in* in_channels*n_out*n_out + in_channels*x[3]*n_out*n_out # DK · DK · M · DF · DF + M · N · DF · DF
          num_multi += mutltis 
          print('* conv %d, DK, M, N, DF = %d %d %d %d, compute %d'%(num_conv, n_in,  in_channels, x[3], n_out, mutltis))
          n_in, j_in, r_in, start_in = n_out, j_out, r_out, start_out
          
          num_conv += 1
          in_channels = x[3]
      else:           # Conv followed by Batchnorm and ReLue 
        layers += [nn.Conv2d(in_channels, out_channels=x[2], kernel_size=(x[0], x[1]), stride=1, padding=(int(x[0]/2), int(x[1]/2)), bias=False), 
                  nn.BatchNorm2d(x[2]),
                  nn.ReLU(inplace=True)] 

        n_out, j_out, r_out, start_out = get_rep_field(n_in, j_in, r_in, start_in, kernel_size=x[0], stride=1, padding=int(x[0]/2)) 
        mutltis = n_in*n_in* in_channels*x[2]*n_out*n_out
        print('* conv %d, DK, M, N, DF = %d %d %d %d, compute %d'%(num_conv, n_in,  in_channels, x[2], n_out, mutltis))
        num_multi +=  n_in*n_in* in_channels*x[2]*n_out*n_out # DK · DK · M · N · DF · DF
        n_in, j_in, r_in, start_in = n_out, j_out, r_out, start_out

        in_channels = x[2]
        num_conv += 1 
    print('* Total Multiplications: %d'%num_multi)
    return nn.Sequential(*layers)


def test():
  net = MobileNet680()
  x = torch.randn(1, 3, 32, 32) 
  y = net(Variable(x)) 
  print(y.size())

if __name__ == "__main__":
  test() 
