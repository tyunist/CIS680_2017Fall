try:
  # CIS 680 nets 
  from .convnet_cis680 import * 
  from .mobilenet_cis680 import * 
  from .resnet_cis680 import * 

  print('Go 1st way') 
except:
  print('Go second way') 
