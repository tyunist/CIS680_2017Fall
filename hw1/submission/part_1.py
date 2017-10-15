from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import time, os
import seaborn as sns
sns.set(color_codes=True)


class SIGMOID():
    def set_weights(self,w):
        self.w = w 
    def set_biases(self,b):
        self.b = b 
    def sigmoid(self,x):
        return 1./(1 + np.exp(-(self.w*x + self.b)))
    
def _L2_loss(y, y_predict):
	return (y - y_predict)**2 

def _grad_L2(y, y_pred, x):
    return 2*(y_pred - y)*y_pred *(1-y_pred)*x

def _cross_entropy_loss(y, y_predict):
    if isinstance(y_predict, np.ndarray):
        y_predict = y_predict + 1e-8*(y_predict==0) - 1e-8*(y_predict==1)
	return -(y*np.log2(y_predict) + (1-y)*np.log2(1-y_predict))

def _grad_cross_entropy(y, y_pred, x):
    return - (y/(y_pred *np.log(2))- (1-y)/( (1-y_pred)*np.log(2))) * y_pred*(1-y_pred)*x

## 
x = 1
w_set = np.linspace(-10, 10, 500)
b_set = np.linspace(-10, 10, 500)

w_set, b_set = np.meshgrid(w_set, b_set)
 
sigmoid_ins = SIGMOID()
sigmoid_ins.set_weights(w_set)
sigmoid_ins.set_biases(b_set)

output_set = sigmoid_ins.sigmoid(x)


fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(w_set, b_set, output_set, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 1)
ax.set_title('Sigmoid Output')
ax.set_xlabel('Weights')
ax.set_ylabel('Biases')
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show() 


## 
y = 0.5 
L2_loss_set = _L2_loss(y, output_set)
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(w_set, b_set, L2_loss_set, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
ax.set_title('L2 Loss')
ax.set_xlabel('Weights')
ax.set_ylabel('Biases')
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show() 


## 
# Analytical gradient 
grad_L2_set = _grad_L2(y, output_set, x)
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(w_set, b_set, grad_L2_set, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
ax.set_title('Gradient of L2 Loss')
ax.set_xlabel('Weights')
ax.set_ylabel('Biases')
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show() 

######################################
# Numerical gradient 
x = 1
y = 0.5 
delta_w = 0.0001
w_set = np.linspace(-10, 10, 500)
b_set = np.linspace(-10, 10, 500)

w_set_left = w_set - delta_w 
w_set_right = w_set + delta_w 
 
w_set_left, _ = np.meshgrid(w_set_left, b_set)
w_set_right, _ = np.meshgrid(w_set_right, b_set)
w_set, b_set = np.meshgrid(w_set, b_set)
sigmoid_ins = SIGMOID()

sigmoid_ins.set_weights(w_set_left)
sigmoid_ins.set_biases(b_set)
output_set_left = sigmoid_ins.sigmoid(x)
L2_loss_left = _L2_loss(y, output_set_left)

sigmoid_ins.set_weights(w_set_right)
output_set_right = sigmoid_ins.sigmoid(x)
L2_loss_right = _L2_loss(y, output_set_right)
num_grad_L2_set = (L2_loss_right -L2_loss_left)/(2.0*delta_w)

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(w_set, b_set, num_grad_L2_set, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
ax.set_title('Numerical Gradient of L2 Loss')
ax.set_xlabel('Weights')
ax.set_ylabel('Biases')
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show() 


## 
y = 0.5 
cross_entropy_loss_set = _cross_entropy_loss(y, output_set)
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(w_set, b_set, cross_entropy_loss_set, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
ax.set_title('Cross-entropy Loss')
ax.set_xlabel('Weights')
ax.set_ylabel('Biases')
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show() 

## 
grad_cross_entropy_set = _grad_cross_entropy(y, output_set, x)
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(w_set, b_set, grad_cross_entropy_set, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
ax.set_title('Gradient of Cross-entropy Loss')
ax.set_xlabel('Weights')
ax.set_ylabel('Biases')
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show() 


######################################
# Numerical gradient 
x = 1
delta_w = 0.0001
w_set = np.linspace(-10, 10, 500)
b_set = np.linspace(-10, 10, 500)

w_set_left = w_set - delta_w 
w_set_right = w_set + delta_w 
 
w_set_left, _ = np.meshgrid(w_set_left, b_set)
w_set_right, _ = np.meshgrid(w_set_right, b_set)
w_set, b_set = np.meshgrid(w_set, b_set)
sigmoid_ins = SIGMOID()

sigmoid_ins.set_weights(w_set_left)
sigmoid_ins.set_biases(b_set)
output_set_left = sigmoid_ins.sigmoid(x)
cross_entropy_loss_left = _cross_entropy_loss(y, output_set_left)

sigmoid_ins.set_weights(w_set_right)
output_set_right = sigmoid_ins.sigmoid(x)
cross_entropy_loss_right = _cross_entropy_loss(y, output_set_right)

num_grad_cross_entropy_set = (cross_entropy_loss_right -cross_entropy_loss_left)/(2.0*delta_w)

fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(w_set, b_set, num_grad_cross_entropy_set, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(0, 1)
ax.set_title('Numerical Gradient of Cross-entropy Loss')
ax.set_xlabel('Weights')
ax.set_ylabel('Biases')
# fig.colorbar(surf, shrink=0.5, aspect=5)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
plt.show() 

