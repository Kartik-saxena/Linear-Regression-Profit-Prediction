function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
x1 = zeros(m,1);
x2 = zeros(m,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
a = zeros(m,1);
a2 = zeros(m,1);
a3 = zeros(m,1);

%for j = 1:m
%x1(j,1) = X(j,1);
%x2(j,1) = X(j,2);
%endfor

x1 = X(:,1);
x2 = X(:,2);


h = X*theta;
t1 = 0;
t2 = 0;

% for i = 1:m
%  a(i,1) = h(i,1) - y(i,1);
%  a2(i,1) = a(i,1).*x1(i,1); 
%  a3(i,1) = a(i,1).*x2(i,1); 
%  endfor   
 
a(:,1) = h(:,1)-y(:,1);
a2 = a(:,1).*x1(:,1);
a3 = a(:,1).*x2(:,1);
 

sumg1= sum(a2);
sumg2= sum(a3);
t1 = theta(1,1)-(alpha/m)*sumg1;
t2 = theta(2,1)-(alpha/m)*sumg2;
   theta(1,1) = t1;
   theta(2,1) = t2;
   
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
