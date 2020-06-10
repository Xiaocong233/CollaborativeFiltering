## Description
Contains a series of MATLAB/Octave Collobrative Filtering Alogorithm functions for constructing machine learning recommender system

## Collobrative Filtering
```
[Ynorm, Ymean] = normalizeRatings(Y, R)
```
  - Preprocess data by subtracting mean rating for every movie (every row)

```
[J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)

```
  - Computes the cost and gradient for the collaborative filtering problem

```
checkCostFunction(lambda)
```
  - Constructs a test collaborative filering problem to check your cost function and gradients
