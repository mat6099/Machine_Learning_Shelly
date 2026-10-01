import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC


#Loading the iris datasets
iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#split into training and tests sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =1, stratify = y ) #stratify used to keep the same class proportions in both training and test data.

#standardize the features
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

#combining training and testing data for plotting
X_combined_std = np.vstack((X_train_std, X_test_std)) #vstack = vertical stacking puts them one above the other.
y_combined = np.hstack((y_train, y_test))         #hstack = horizontal stacking puts them side by side.



#create an svm classifier
svm = SVC(kernel = 'linear', C = 1.0, random_state = 1)  #svc = support vector classifier, we are calling out the svm clf here. kernel = 'linear' for linear SVM, C = 1.0 is the regularization parameter.

#train the svm
svm.fit(X_train_std, y_train)

#plot the decision regions
plot_decision_regions(X_combined_std, y_combined, clf = svm )
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()
