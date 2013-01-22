% L-BFGS demo

%% Preprocess data
load('data/USPSN_normalized_digit0.mat');

X = data(:,1:end-1);    % (nExamples,nFeatures)
y = data(:,end);        % (nExamples,1)

nExamples = size(X,1);
nFeatures = size(X,2);

w = zeros(nFeatures,1); % (nFeatures,1)

options.Display = 0;

%% Logistic Regression with L2 Regularization
% Find L2-regularized logistic
funObj = @(w)LogisticLoss(w,X,y);
lambda = 10*ones(nFeatures+1,1);
lambda(1) = 0; % Don't penalize bias
fprintf('Training L2-regularized logistic regression model...\n');
wL2 = minFunc(@penalizedL2,zeros(nFeatures+1,1),options,funObj,lambda);

trainErr_L2 = sum(y ~= sign(X*wL2))/length(y)

% Plot the result
figure;
plotClassifier(X,y,wL2,'MAP Logistic');
fprintf('Comparison of norms of parameters for MLE and MAP:\n');