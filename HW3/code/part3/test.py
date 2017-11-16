import torch
from  torch.autograd import Variable
import numpy as np
import pdb  
if __name__=='__main__':
  pdb.set_trace()
  x = Variable(torch.from_numpy(0.5 * np.arange(2*3*36).reshape(2, 3, 36)))
  x_p = x.permute(2,0,1)
  idx = torch.LongTensor([i for i in range(0,20,2)])
  y = x_p[idx,:,:]
  y_final = y.permute(1,2,0)
    
