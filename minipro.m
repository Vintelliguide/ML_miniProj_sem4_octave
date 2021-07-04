
clear ; close all; clc



data = load('minidatatrain.csv');
X = data(:, [1, 2, 3, 4, 5]); y = data(:, 6);




% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros) \n');
grad

fprintf('\nProgram paused. Press enter to continue.\n');
pause;




% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 3;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);


% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);




data = load('minidatatest.csv');
X = data(:, [1, 2, 3, 4, 5]); y = data(:, 6);


p = predict(theta, X);

fprintf('Test Accuracy: %f\n', mean(double(p == y)) * 100);
