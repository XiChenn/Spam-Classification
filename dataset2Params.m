function [C, sigma] = dataset2Params(X, y, Xval, yval)
%This function returns your choice of C and sigma the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% You should use this function to try all the different combinations of learning
% parameters. One way to do this is to write a double-for loop to try all
% the different C and sigma combinations and keep the ones that
% give you the lowest error. HOWEVER for your project submission, for full
% credit you should comment out the for-loop code and only do the calculations
% using your final results (otherwise it will take your professor forever to
                            % grade the projects)
%
%  Hint: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
lowestError = inf;
paraVec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%for i = 1:length(paraVec)
%    C = paraVec(i);
%    for j = 1:length(paraVec)
%        sigma = paraVec(j);
%	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
%	predictions = svmPredict(model, Xval);
%	error = mean(double(predictions ~= yval));
%	if (error < lowestError)
%	    lowestError = error;
%	    bestC = C;
%	    bestSigma = sigma;
%	endif
%    end
%end

bestC = 1; % Found by the commented loop
bestSigma = 0.1; % Found by the commented loop
model= svmTrain(X, y, bestC, @(x1, x2) gaussianKernel(x1, x2, bestSigma)); 
predictions = svmPredict(model, Xval);
lowestError = mean(double(predictions ~= yval));

% =========================================================================
% Save and print the values you are returning along with the lowest error value 
C = bestC
sigma = bestSigma
lowestError

end

