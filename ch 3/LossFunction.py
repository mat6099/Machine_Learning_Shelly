import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_1(z):
    return - np.log(sigmoid(z))
def loss_0(z):
    return - np.log(1-sigmoid(z))

z = np.arange(-10,10,0.1)
sigma_z = sigmoid(z)
cl = [loss_1(x) for x in z]   # to calculate the loss function for y=1
plt.plot(sigma_z, cl, label='L(w,b) if y = 1')
c0 = [loss_0(x) for x in z]    # to calculate the loss function for y=0
plt.plot(sigma_z, c0, linestyle = '--', label = 'L(w,b) if y = 0')
plt.ylim(0.0, 5.1)
plt.xlim([0,1])
plt.xlabel('$\sigma(z)$')
plt.ylabel('L(w,b)')
plt.legend(loc = 'best') # to set the legend at the best location
plt.tight_layout()       # to automatically adjust the subplot parameters to give specified padding
plt.show()