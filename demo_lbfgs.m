% Project 1
% CSE 250B
% Winter 2013


%% Load data
load('data/Web_normalized.mat');


%% Preprocess data
mTest = data(1:100,:);
mValid= data(100:200,:);
mu = 0.1;

%% Run L-BFGS
addpath('minFunc');

[V,T,Var] = LBFGS(mTest, mValid, mu);


%% Plot results
