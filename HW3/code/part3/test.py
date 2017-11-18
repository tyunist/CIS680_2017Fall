 
# from  torch.autograd import Variable
import numpy as np
import pdb  
import matplotlib.pyplot as plt 
min_lr = 1e-4 
lr = 1e-3 
max_epoches = 20 

lr_decay_f = (min_lr/lr)**(1./(max_epoches-1) )

def adjust_lr(cur_lr, lr_decay_f):
	lr = cur_lr*lr_decay_f 
	print('===> cur_lr %.5f, updated lr %.5f, decay factor %.4f'%(cur_lr, lr, lr_decay_f))
	return lr 
lr_array = []
epoches = []
for i in range(max_epoches):
	epoches.append(i)
	lr_array.append(lr)
	lr = adjust_lr(lr, lr_decay_f)

plt.plot(lr_array)
plt.show()  
