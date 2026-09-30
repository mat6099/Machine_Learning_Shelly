import numpy as np 
import matplotlib.pyplot as plt 
 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from mlxtend.plotting import plot_decision_regions 
 
# Loading the iris data, only first two classes and two features 
iris = load_iris() 
X = iris.data[:, [2, 3]] # first 100 rows and 2 columns sepal length and petal length 
y = iris.target[:] # class labels being 0 or 1 
 
#split into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =1, stratify=y) 
 
#standardize the features 
sc = StandardScaler() 
 
X_train_std = sc.fit_transform(X_train) 
X_test_std = sc.transform(X_test) 
 
 
#define logistic regression using gradient descent 
class LogisticRegressionGD: 
 
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1): 
        self.eta = eta 
        self.n_iter = n_iter 
        self.random_state = random_state 
 
    def fit(self, X, y): 
        rgen = np.random.RandomState(self.random_state) 
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1]) 
        self.b_ = 0.0 
        self.losses_ = [] 
 
        for i in range(self.n_iter):  #gradient descent
            net_input = self.net_input(X) 
            output = self.activation(net_input) 
            errors = (y - output) 
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0] 
            self.b_ += self.eta * 2.0 * errors.mean() 
            loss = (-y.dot(np.log(output))-((1-y).dot(np.log(1 - output))))/X.shape[0]
            self.losses_.append(loss) 
        return self 
 
    def net_input(self, X): 
        return np.dot(X, self.w_)+ self.b_ 
 
 
    def activation(self, z): 
        return 1.0/ (1.0 + np.exp(-np.clip(z, -250, 250))) 
 
    def predict(self, X): 
        return np.where (self.activation(self.net_input(X)) >=0.5, 1, 0) 
 
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)] 
y_train_01_subset = y_train[(y_train ==0) | (y_train ==1)] 
 
 
# train the model 
lrgd = LogisticRegressionGD(eta = 0.3, n_iter = 1000, random_state =1) 
lrgd.fit(X_train_01_subset, y_train_01_subset) 
 
 
#plot the decision regions using standardized training data 
plot_decision_regions(X= X_train_01_subset, y = y_train_01_subset, clf = lrgd) 
plt.xlabel('Petal length [standardized]') 
plt.ylabel('Petal width [standardized]') 
plt.legend(loc = 'upper left') 
plt.tight_layout() 
plt.show()  