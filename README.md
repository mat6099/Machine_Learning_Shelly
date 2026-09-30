# Chapter 3 : A Tour of Machine Learning Classifiers Using Scikit-Learn

## Introduction to Classification
- **Classification** is a supervised learning task in which a model learns to assign data points to categories or classes.
- Different algorithms have different strengths, so the choice of classifier depends on the dataset and the problem.
- A general supervised learning workflow:
  1. Select the features and collect labelled data.
  2. Choose a performance metric.
  3. Select and train a classifier.
  4. Evaluate its performance.
  5. Tune or improve the model if needed.

## First steps with scikit-learn - Training a Perceptron
- scikit-learn is a Python library with machine learning algorithms and tools.
- We took iris dataset : 150 flowers, 3 species (with labels 0,1,2) and 2 measured features i.e. petal length and petal width as input features (x) and species labels as targets (y).
- We divide the data into training set and a test set where 30% data was for testing and remaining were training examples i.e. 105 training examples and 45 test examples. We also stratify the data such that the split is made into training and test data so each set keeps roughly the same proportion of each class label as the original data set.
- Next we standardize the features which helps in putting the features on a comparable scale. $z = \frac{x-\mu}{\sigma}$ is used to standardize where $\mu$ is the mean and $\sigma$ is the standard deviation. Learning scaling parameters from the training data, and then applying the same transformation to test data.
- After that we train and evaluate a perceptron. fit() trains the model using features and their correct labels.
- In the end , we use the model to predict labels for the test examples and compare predictions with true labels to evaluate performance.

## Modelling Class Probabilities via Logistic Regression
- The disadvantage of the perceptron is that it never converges if the classes are not perfectly linearly separable.
- Logistic Regression is a classification model that performs very well on linearly separable classes.
- First, the model calculates a score (net input):

  $$z = \mathbf{w}^T\mathbf{x} + b$$

  where:
  - $\mathbf{x}$: input features
  - $\mathbf{w}$: weights learned by the model
  - $b$: bias

### Odds
- Let $p$ be the conditional probability that a particular example belongs to class 1 given its features $\mathbf{x}$:

  $$p = P(y=1|\mathbf{x})$$

- The odds in favor of the positive event are:

  $$\frac{p}{1-p}$$

### Logit
- The logit is the natural logarithm of the odds:

  $$logit(p)
  = \log\left(\frac{p}{1-p}\right)$$

- The logit function transforms probability values in the range $(0,1)$ into values over the entire real-number range $(-\infty,\infty)$.

- In Logistic Regression, we assume a linear relationship between the log-odds and the net input:

  $$logit(p)=\mathbf{w}^T\mathbf{x}+b = z$$ 

### Logistic Sigmoid Function
The inverse of the logit function is called the logistic sigmoid function.
It has a characteristic S-shape:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Here, $z$ is the net input.
The sigmoid function transforms real-number inputs into values in the range $[0,1]$.
At $z=0$:

$$
\sigma(0) = 0.5
$$

The output of the sigmoid function is interpreted as the probability
that an example belongs to class 1:

$$
\sigma(z) = P(y=1 \mid \mathbf{x})
$$

### Likelihood
Likelihood measures how well the model's predicted probabilities fit
the actual training labels.
Let $ y $ be the actual label (0 or 1), and $p$ be the predicted
probability of class 1. The likelihood contribution is:

$$
p^y(1-p)^{1-y}
$$

- If $y=1$, the contribution is $p$.
- If $y=0$, the contribution is $1-p$.

Likelihood is the probability the model assigned to the class that actually occured.

## Converting an Adaline implementation into an algorithm for logistic regression.
Adaline classifier can be turned into a working Logistic Regression classifier by changing just two things:
1. Activation function: from identity to sigmoid function
2. Loss function: from sum of squared errors to logistic loss
Everything else remains the same.

## Tackling Overfitting via Regularization
### Overfitting
- When model performs very well on training data but poorly on test data.
- Means model has learned training data too closely and does not generalize well.
- Often associated with model that is too complex.
### Regularization
- Technique used to reduce overfitting.
- Adds a penalty for larger weights to loss function.
- Encourages a simpler model even though the training loss increases slightly.
#### L1 Regularization

$$ \lambda \sum_{j}|w_{j}| $$
- Some weights can become exactly 0.
- Can therefore perform feature selection since w = 0 for those .

#### L2 Regularization

$$ \frac{\lambda}{2} \sum_{j} {w_{j}^2} $$
- Penalises strong weights strongly. Working reducing the model to simpler model.
- Shrinks them to 0 but does not make them 0.

#### C in Logistic Regression
- small $C$ $\implies$ strong regularization
- large $C$ $\implies$ weak regularization

## Maximum Margin Classification with Support Vector Machines

### Margin
The distance between the decision boundary and the closest training examples.

### Support Vectors
The training examples closest to the decision boundary. They determine the maximum margin boundary.

### SVM
It finds a decision boundary that maximises the margin between the classes. A larger margin tends to give better generalization , while a smaller margin can be more prone to overfitting.
#### Soft Margin SVM
Real datasets are not perfectly linear separable. So SVM allows some violations, allows the slack variables ( the training examples closest to the decision boundary) to lie inside the margin or even be misclassified.

### Role of C
- Large C $\implies$ strong penalty for violations i.e. tries harder to classify training examples correctly.
- Small C $\implies$ weak penalty for violations i.e. allows a wider margin.


## Solving Non-Linear problems using a Kernel SVM

### Kernel Methods
- Linear SVM works well when classes can be separated by a straight hyperplane.
- For non linearly separable data (e.g. XOR), a linear SVM cannot find a suitable decision boundary.
- Kernel methods creates non linear combinations of the original features to project them onto a higher-dimensional space via a mapping function, $\phi$, where the data becomes linearly separable.
- Example:
  $$
\phi(x_1, x_2) = (x_1, x_2, x_1^2 + x_2^2) $$

### Kernel Trick :
- Explicitly calculating the high-dimensional mapping $\phi$ can be computationally expensive.
- Kernel trick allows us to work as if the data were mapped into a higher-dimensional space without explicitly calculating that mapping.
- Conceptually:
  $$ K(x_i, x_j) = \phi (x_i)^T \phi(x_j) $$

### RBF Kernel
- Radial Basis Function is a commonly used kernel for non linear SVM. It allows SVM to create nonlinear decision boundaries.

#### Gamma 
- $\gamma $ controls the influence of individual training examples in an RBF SVM.
- Small $\gamma$ $\implies$ broader influence $\implies$ smoother/softer decision boundary.
- Large $\gamma$ $\implies$ more localised influence $\implies$ potentially more complex decision boundary.
