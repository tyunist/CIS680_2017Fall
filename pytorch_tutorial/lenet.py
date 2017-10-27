'''LeNet in PyTorch.'''
import torch,numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable

class LeNet(nn.Module):
  def __init__(self):
    super(LeNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 5, stride = 1, padding = 2)
    self.conv2 = nn.Conv2d(32, 32, 5, stride = 1, padding = 2)
    self.conv3 = nn.Conv2d(32, 64, 5, stride = 1, padding = 2)
    self.conv4 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
    self.conv5 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
    self.conv6 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
    self.conv7 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
    self.conv8 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
    self.conv9 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
    self.fc1   = nn.Linear(1024, 64)
    self.fc2   = nn.Linear(64, 10)
    self.fc3   = nn.Linear(84, 10)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    out = F.avg_pool2d(x, 2)
    out = F.relu(self.conv2(out))
    out = F.avg_pool2d(out, 2)
    pdb.set_trace()
    out = F.relu(self.conv3(out))
    out = F.avg_pool2d(out, 2)
    out = out.view(out.size(0), -1)
    out = F.relu(self.fc1(out))

    out = self.fc2(out)
    return out

model = LeNet()
a = Variable(torch.from_numpy(np.ones((10,3,32,32)).astype(np.float32)))
pred = model(a)

