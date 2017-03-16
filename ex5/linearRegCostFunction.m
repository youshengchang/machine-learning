function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
h = X * theta;
J = (h  - y)' * (h -y)/(2 * m);
theta = theta(2:end);
reg = (theta' * theta)* (lambda/(2 * m));
J = J + reg;
grad(1) = (X(:, 1)'* (h -y))/m;
%grad(2:size(theta)) = (X(:,2:size(X)(:,2))' * (h - y))/m + (lambda * theta)/m;
grad(2:end) = (X(:,2:end)' * (h - y))/m + (lambda/m) * theta;












% =========================================================================

grad = grad(:);

end
