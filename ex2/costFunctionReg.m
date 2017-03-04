function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h  = sigmoid( X * theta);
a1 = y' * log(h) * (-1);
a2 = ((1 - y)' * log(1 - h)) * (-1);
J = (a1 + a2)/m;
theta1 = theta(2:size(theta));
reg = ((theta1' * theta1) * lambda)/(2 * m);
J = J + reg;
grad(1) = (X(:,1)' * (h - y))/m;
grad(2:size(theta)) = (X(:,2:size(X)(:,2))' * (h - y))/m + (lambda/m)* theta1;






% =============================================================

end
