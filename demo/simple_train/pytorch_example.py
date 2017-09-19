import torch
from torch.autograd import Variable

# Model parameters
W = Variable(torch.FloatTensor([.3]), requires_grad=True)
b = Variable(torch.FloatTensor([-.3]), requires_grad=True)
# Model input and output
x = Variable(torch.FloatTensor([1, 2, 3, 4]), requires_grad=False)
linear_model = W * x + b
y = Variable(torch.FloatTensor([0, -1, -2, -3]), requires_grad=False)

for i in range(1000):
  linear_model = W * x + b
  # loss
  loss = (linear_model - y).pow(2).sum() # sum of the squares
  loss.backward()

  W.data -= 0.01 * W.grad.data
  W.grad.data.zero_()
  b.data -= 0.01 * b.grad.data
  W.grad.data.zero_()

# evaluate training accuracy
print("W: %s\nb: %s\nloss: %s"%(W.data, b.data, loss.data))

