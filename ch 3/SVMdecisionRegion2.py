import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

#height and weights as dataset
X = np.array([[160,55], [170,65], [165,60], [180,75], [181,80], [173,70], [160,55], [178,75],[175,72], [182,85], [168,62], [177,78]])
y = np.array([0,0,0,1,1,1,0,1,1,1,0,1])   # class 1 for females and class 0 for males

svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)   # kernel = 'linear' for linear SVM, C = 1.0 is the regularization parameter
svm.fit(X, y)

#plotting decision regions
plot_decision_regions(X,y, clf = svm)
plt.xlabel('height(cm)')
plt.ylabel('weight(kg)')
plt.title('SVM Decision Region for Height and Weight')
plt.show()
