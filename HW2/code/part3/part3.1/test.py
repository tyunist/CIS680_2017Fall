import numpy as np 

a = np.array([[0, 1, 2],   [3, 4, 5], [2,4,6], [3,5,8]])


print np.where(a == np.max(a))[0][0]
