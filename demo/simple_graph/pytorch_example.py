import numpy as np
np.random.seed(0)
import torch
from torch.autograd import Variable

n1, n2, n3 = 8, 4, 2

# Computation graph
x = Variable(torch.from_numpy(np.random.randn(n1, n2)).float(),\
     requires_grad=True)
y = Variable(torch.from_numpy(np.random.randn(n2, n3)).float(),\
     requires_grad=True)
z = Variable(torch.from_numpy(np.random.randn(n1, n3)).float(),\
     requires_grad=True)

a = torch.matmul(x, y)
b = a + z
c = torch.mean(b)

# Calculate gradients
c.backward(torch.ones(1))
grad_x, grad_y, grad_z = x.grad, y.grad, z.grad

# Print results
print("Output c = %f" % c.data.numpy())
print("Mean gradients of (x, y, z) = (%f, %f, %f)" % \
  (np.mean(grad_x.data.numpy()), \
   np.mean(grad_y.data.numpy()), np.mean(grad_z.data.numpy())))

