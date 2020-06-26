## Description
Contains a series of MATLAB/Octave Collobrative Filtering functions and for constructing different machine learning recommender systems

## Collobrative Filtering
```
[Ynorm, Ymean] = normalizeRatings(Y, R)
```
  - Preprocess data by subtracting mean rating for every movie (every row)
___

```
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)

```
  - Computes the cost and gradient for the collaborative filtering problem
___

```
checkCostFunction(lambda)
```
  - Constructs a test collaborative filering problem to check your cost function and gradients
