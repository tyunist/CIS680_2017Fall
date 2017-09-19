import numpy as np
np.random.seed(0)

n1, n2, n3 = 8, 4, 2

# Computation graph
x = np.random.randn(n1, n2)
y = np.random.randn(n2, n3)
z = np.random.randn(n1, n3)

a = np.matmul(x, y)
b = a + z
c = np.mean(b)

# Calculate gradients
grad_c = 1.0 
grad_b = grad_c * np.ones([n1, n3]) / (n1 * n3)
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = np.matmul(grad_a, y.transpose())
grad_y = np.matmul(x.transpose(), grad_a)

# Print results
print("Output c = %f" % c)
print("Mean gradients of (x, y, z) = (%f, %f, %f)" % \
  (np.mean(grad_x), np.mean(grad_y), np.mean(grad_z)))

