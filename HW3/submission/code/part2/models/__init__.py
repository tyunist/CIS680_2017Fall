try:
  # CIS 680 nets 
  from .convnet_cis680 import * 
  from .mobilenet_cis680 import * 
  from .resnet_cis680 import * 
  from .faster_rcnn_cis680 import * 

  from .faster_rcnn_mobile_cis680 import * 
  from .faster_rcnn_res_cis680 import * 
  from .spatial_transformer import *
  print('Go 1st way') 
except:
  print('Go second way') 
  from spatial_transformer import * 
