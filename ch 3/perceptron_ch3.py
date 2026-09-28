from sklearn import datasets
import numpy as np

#loading iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

print('Class labels:', np.unique(y))

#Creating training and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

print('Labels counts in y:', np.bincount(y))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test:', np.bincount(y_test))

#Standardizing the features
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Training a perceptron model
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state = 1)
ppn.fit(X_train_std,y_train)

#Performing predictions on the test dataset
y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' %(y_test != y_pred).sum())

#Calculating the accuracy of the model
from sklearn.metrics import accuracy_score
print('Accuracy : %.3f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.3f' %ppn.score(X_test_std, y_test))

#Plotting the decision regions
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib import colors
def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
    markers = ('o','s','^','v','<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

#plotting the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha = 0.3, cmap= cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl,0],
                    y = X[y == cl, 1],
                    alpha = 0.8,
                    c = colors[idx],
                    marker = markers[idx],
                    label=f'Class {cl}',
                    edgecolor = 'black')

#highlighting test examples
    if test_idx is not None:
    #plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0],
                X_test[:, 1],
                c = 'none',
                edgecolor = 'black',
                alpha = 1.0,
                linewidth = 1,
                marker = 'o',
                s = 100,
                label = 'Test set')

#plotting the decision regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y = y_combined,
                      classifier = ppn,
                      test_idx = range(105,150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()