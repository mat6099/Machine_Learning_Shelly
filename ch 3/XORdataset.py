import matplotlib.pyplot as plt
import numpy as np

#fix the random numbers so that we get the same dataset every time
np.random.seed(1)
X_xor = np.random.randn(200, 2)  #generate 200 random datapoints with 2 features
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:,1] > 0) #xor gives true when exactly one of the two conditions is true
y_xor = np.where(y_xor, 1, 0) #true is replaced with 1 and false is replaced with 0. This is done to make it compatible with the SVM classifier which expects the labels to be in 0 and 1 format.

#plotting the dataset. y_xor == 1 means the points belonging to class 1 of column 0 or 1 and y_xor == 0 means the points belonging to class 0.
plt.scatter(X_xor[y_xor == 1,0], X_xor[y_xor ==1, 1], c = 'royalblue', marker = 's', label = 'class 1') #plot points belonging to class 1 # 
plt.scatter(X_xor[y_xor == 0,0], X_xor[y_xor ==0, 1], c = 'tomato', marker = 'o', label = 'class 0') # plot points belonging to class 0
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('XOR Dataset')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()