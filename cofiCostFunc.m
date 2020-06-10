% Computes the cost and gradient for the collaborative filtering problem
function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)

  % X - num_movies  x num_features matrix of movie features
  % Theta - num_users  x num_features matrix of user features
  % Y - num_movies x num_users matrix of user ratings of movies
  % R - num_movies x num_users matrix, where R(i, j) = 1 if the 
  % i-th movie was rated by the j-th user

  % Unfold the U and W matrices from params
  X = reshape(params(1:num_movies*num_features), num_movies, num_features);
  Theta = reshape(params(num_movies*num_features+1:end), ...
                  num_users, num_features);
      
  % You need to return the following values correctly
  J = 0;
  X_grad = zeros(size(X));
  Theta_grad = zeros(size(Theta));

  % Compute for the unregularized cost (Known bug in MATLAB mistakening conditional indexing)
  J = 1 / 2 * sum(sum(((X * Theta' - Y)(R == 1) .^ 2)));
  % Adding regularization unit
  J = J + lambda / 2 * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));

  % Compute for regularized x gradients
  for i=1:num_movies
    idxu = find(R(i, :) == 1);
    X_grad(i, :) = (X(i, :) * Theta(idxu, :)' - Y(i, idxu)) * Theta(idxu, :) ...
                    + lambda * X(i, :);
  endfor

  % Compute for regularized theta gradients
  for j=1:num_users
    idxm = find(R(:, j) == 1);
    Theta_grad(j, :) = (X(idxm, :) * Theta(j, :)' - Y(idxm, j))' * X(idxm, :) ...
                        + lambda * Theta(j, :);
  endfor

  grad = [X_grad(:); Theta_grad(:)];

end
