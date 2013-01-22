%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tlogistic_test.m       tlogistic_test

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function

% tlogistic_test: the test algorithm of t-logistic regression


% input

% theta:  the output of tlogistic_train, model parameter theta
% x_test: \phi(x_train) in DxN, where N is #sample, D is #dimension
% y_test: labels y in 1xN


% output

% error:   the test error rate

%%

function error=tlogistic_test(theta,x_test,y_test)

score=theta'*x_test;
yt=sign(score);
error=sum(yt~=y_test)/length(y_test);
