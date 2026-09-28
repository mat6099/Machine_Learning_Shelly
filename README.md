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

## Modelling class probabilities via Logistic Regression
- The disadvantage of perceptron is that it never converges if the classes are not perfectly linearly separable.
- Logistic Regression is a classification model that works well on linearly separable classes.
- First , the model calculates a score: $$\z = \mathbf{w}^(T) + b $$ where x: input features\\ w : weights learned by the model\\ b: bias
- ### Odds:
- Let $p$ be the conditional probability that a particular example belongs to certain class 1 given its features, x. Then we define odds: the odds in favor of a particular event. The odds is defined as $$\frac{p}{(1-p)}$$ where p stands for probability of the positive event.
- ### Logit:
- It is simply the natural logarithm of the odds. $$logit(p) = \log \frac{p}{(1-p)} $$. The logit function takes the input values in the range 0 to 1 and transforms them into values over the entire real number range.
- ### Logistic sigmoid function:
- This inverse of the logit function is called logistic sigmoid function, which is called sigmoid function due to its characteristic S shape. It is defined as: $$ \sigma(z) = \frac{1}{1+\exp(-z)} $$ where, z is the net input . This sigmoid function takes real number values as input and transforms them into values in the range $[0, 1]$ with an intercept at $\sigma(0) = 0.5 $. This function acts as the activation function for Logistic Regression.
- The predicted probability can then simply be converted into a binary outcome via a threshold function \[
\hat{y} =
\begin{cases}
1, & \text{if } z \geq 0,\\
0, & \text{if } z < 0.
\end{cases}
\]
- Logistic Regression has wide applications in the field of medicine. It is also used in weather forecasting.
- ## Learning the model weight via the Logistic loss function.
- ### Likelihood 
- 
- 
