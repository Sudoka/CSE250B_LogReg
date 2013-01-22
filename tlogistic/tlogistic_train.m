%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tlogistic_train.m       tlogistic_train

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function

% tlogistic_train: the training algorithm of t-logistic regression


% input

% x_train: \phi(x_train) in DxN, where N is #sample, D is #dimension
% y_train: labels y in 1xN
% t:       choose from 1.1, 1.2, ..., 1.9 in order to use 'gpoints.mat'
% lambda:  parameter for the impact of the regularizer (not lambdat)
% iter_xi: iterations of xi-step
% iter_theta: iterations of theta-step
% theta_init: initialization of theta

% output

% theta:   the output model parameter
% xi:      the impact of each z_n, first D element are prior, last N are
%          loss from the training data

%%

function [theta,xi]=tlogistic_train(x_train,y_train,t,lambda, iter_xi, iter_theta, theta_init)

addpath ../minFunc  % this is the directory of the optimizer we use in the \theta-step 

load('gpoints.mat');  % this is the pre-computed g(u) for exp_t(u-g(u)) at certain points of u, use spline method to compute g(u) in general
global pp;
pp=polyp;

[D,N]=size(x_train);

theta=theta_init;
v=2/(t-1)-1;  % convert to degree of freedom for student's t-prior
Psi=(gamma((v+1)/2)*sqrt(lambda/2)/sqrt(pi*v)/gamma(v/2))^(-2/(v+1));
lambdat=lambda*Psi/(t-1)/v;
gt=(Psi-1)/(t-1);

niter=iter_xi;   % # of xi-step (outer loop)

options.Method = 'lbfgs';
options.Display='off';
options.MaxFunEvals= iter_theta;   % # of theta-step (inner loop)

for iter=1:niter
    
    % xi-step
    z=t_Loss_EM(theta,x_train,y_train,t,lambdat,gt);
    xi=1./z;  % xi is proportional to 1/z
    
    % theta-step
    thetat=minFunc(@t_Lossgrad_EM,theta,options,xi,x_train,y_train,t,lambdat,gt);    
    
    % stop criteria
    if norm(thetat-theta)/norm(theta)<1e-4
        theta=thetat;break;
    end
    theta=thetat;
end
xi=xi(D+1:end);  % only return \xi of the training data

