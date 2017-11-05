# More, refer to: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py 
import torch.nn.functional as F # have no trainable params 
import torch.nn as nn  # have trainable params 
import nump as np 
import torch.utils.data as data
import cPickle as pickle  
class cifar_10_model(nn.Module):
	def __init__(self):
		# pre-define layers here 
		self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2) # in_channels, out_channels, kernel_size
		self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
		self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
		self.fc1   = nn.Linear(8*8*64, 64)      # in_features, out_features, bias=True
		self.fc2   = nn.Linear(64, 10)
		self.bn1   = nn.BatchNorm2d(32)
		self.bn2   = nn.BatchNorm2d(32)
		self.bn3   = nn.BatchNorm2d(64)
		self.bn4   = nn.BatchNorm2d(64)

	def forward(self, x):
		# Do forward computation 
		x = self.conv1(x)
		x = F.relu(self.bn1(x))
		x = F.avg_pool2d(x, 2)
		x = self.conv2(x)
		x = F.relu(self.bn2(x))
		x = F.avg_pool2d(x, 2)
		x = self.conv3(x)
		x = F.relu(self.bn3(x))
		x = F.avg_pool2d(x, 2)

		# Reshape x from 4D to 2D, before reshape, x.size()_= N*C*H*W. After reshaping, x.size() = N*D 
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		x = F.relu(self.bn4(x))
		x = self.fc2(x)

		return  x 

class dataloader_obj(data.Dataset):
	def __init__(self, param):
		f_data = open(param['data_path'], 'rb')
		self.data = pickle.load(f_data)
		# reshape to N*C*H*W 
		self.data = self.data.reshape(self.data.shape[0], 3, 32, 32)
		f_data.close()

		f_label = open(param['data_label'], 'rb')
		self.label = pickle.load(f_label)
		f_label.close()
		self.mean_val = param['mean_val']
		self.std = param['std']

	def __getitem__(self, index):
		#
		img = self.data[index, :, :, :][np.newaxis,...]
		print '>> shape of img:', img.shape 

		# Normalize image 
		img = (img - self.mean_val.reshape(1, 3, 1, 1))/(self.std.reshape(1, 3, 1, 1))
		label = self.label[index]

		# convert numpy array to torch tensor variable 
		img = Variable(torch.from_numpy(img.astype(np.float32)))
		label = Variable(torch.from_numpy(label)).type(torch.LongTensor)
		return img, label 

	def __len__(self):
		return self.data.shape[0]


def run():
	# initialize the defined model 
	model = cifar_10_model()
	dataloader_obj = dataloader_obj(param)

	trainloader = data.DataLoader(dataloader_obj, batch_size=100, shuffle=True, num_workers=2)
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
	# define loss function 
	loss_fn = nn.CrossEntropyLoss()

	# Set model to train 
	model.train() 
	for batch_idx, (img_data, img_label) in enumerate(trainloader):
		# Feed input  image to model and do the forward computation and get the result 
		pred = model.forward(img_data)
		# compute the loss 
		loss = loss_fn(pred, img_label) 

		# Remember to use optimizer.zero_grad() every time before computing gradients. 
		# Reason: the grad will accumulate and will not automatically cleaned after each iteration.
		optimizer.zero_grad()
		# Compute grad for all parameters related to the loss 
		loss.backward()
		# Update the parameter based on the grad 
		optimizer.step() 




# import numpy as np 
# import matplotlib.pyplot as plt 
# import time 
# beta1_array = np.arange(0.8, 0.9, 0.05)
# beta2_array = np.arange(0.9, 0.999, 0.01)
# n_beta1 = beta1_array.shape[0]
# n_beta2 = beta2_array.shape[0]
# lr_array = np.zeros((n_beta1, n_beta2, 10))
# lr = 0.1
# i  = 0 
# print beta1_array.shape[0]
# print beta2_array.shape[0]
# time.sleep(2)
# for beta1 in beta1_array:
# 	j = 0 
# 	for beta2 in beta2_array:
# 		print 'beta1, beta2:', beta1, beta2 
# 		print  'i, j:', i, j 
# 		for t in range(10):
# 			lr_tmp = lr * np.sqrt(1 - np.power(beta2,t)) / (1 - np.power(beta1,t))
# 			print lr_tmp
# 			lr_array[i, j, t] = lr_tmp
		
# 		plt.plot(lr_array[i, j, :])
# 		plt.legend(str(i) + ' & ' + str(j))
# 		j+= 1 
# 	i+= 1



# plt.show()