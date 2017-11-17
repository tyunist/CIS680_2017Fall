try:
  from .vgg import *
  from .dpn import *
  from .lenet import *
  from .senet import *
  from .resnet import *
  from .resnext import *
  from .densenet import *
  from .googlenet import *
  from .mobilenet import *
  from .shufflenet import *
  from .preact_resnet import *
  # CIS 680 nets 
  from .convnet_cis680 import * 
  from .mobilenet_cis680 import * 
  from .resnet_cis680 import * 
  from .faster_rcnn_cis680 import * 
  from .spatial_transformer import *
  print('Go 1st way') 
except:
  print('Go second way') 
  from spatial_transformer import * 
