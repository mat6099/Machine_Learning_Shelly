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

  $$\operatorname{logit}(p)
  = \log\left(\frac{p}{1-p}\right)$$

- The logit function transforms probability values in the range $(0,1)$ into values over the entire real-number range $(-\infty,\infty)$.

- In Logistic Regression, we assume a linear relationship between the log-odds and the net input:

  $$\operatorname{logit}(p)=\mathbf{w}^T\mathbf{x}+b$$

### Logistic Sigmoid Function
- The inverse of the logit function is called the logistic sigmoid function.
- It has a characteristic S-shape:

  $$\sigma(z)=\frac{1}{1+\exp(-z)}$$

- Here, $z$ is the net input.

- The sigmoid function transforms real-number inputs into values in the range $[0,1]$.

- At $z=0$:

  $$\sigma(0)=0.5$$

- The output of the sigmoid function is interpreted as the probability that an example belongs to class 1:

  $$\sigma(z)=P(y=1|\mathbf{x})$$
- Logistic Regression has wide applications in the field of medicine. It is also used in weather forecasting.
- ## Learning the model weight via the Logistic loss function.
- ### Likelihood
- Likelihood measures how well the model's current predicted probabilities fit the actual training labels. For example, let (y) be the actual label(0 or 1), and (p) be the predicted probability of class 1. Its likelihood contribution is: $ p^y(1-p)^{1-y} $
- 1. if (y=1), the contribution is (p).
  2. if (y=0), the contribution is (1-p).
- So the contribution is the probability the model is assigned to the correct class.
- 
- 
