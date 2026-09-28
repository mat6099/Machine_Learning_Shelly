import matplotlib.pyplot as plt
import numpy as np

#defined sigmoid function
def sigmoid(z):
                  return 1/(1+np.exp(-z))
z = np.arange(-7,7,0.1)
sigma_z = sigmoid(z)
plt.plot(z, sigma_z)
plt.axvline(0.0, color = 'k')  #a vertical line at x=0
plt.ylim(-0.1, 1.1)             # to set space after 0 and 1 on y-axis
plt.xlabel('z')
plt.ylabel('$\ sigma (z)$')

plt.yticks([0.0, 0.5, 1.0])     #to set the y-ticks at 0, 0.5 and 1.0
ax = plt.gca()                   # to get the current axis
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()