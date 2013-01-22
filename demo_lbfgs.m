% Project 1
% CSE 250B
% Winter 2013


%% Load data
load('data/Web_normalized.mat');


%% Preprocess data
x = data(1:10,1:end-1);
y = data(1:10,end);

% initialize
nFeatures = size(x,2);
B0 = zeros(1,nFeatures); % parameters

% function handle
fhandle = @(B) derivativeLCL(x,y,B);

%% Test function
%meanDLCL = derivativeLCL(x, y, B0)


%% Setup L-BFGS
options.Method = 'lbfgs';
options.Dispaly = 'iter';
options.MaxFunEvals = 1E3;

% add minFunc directory to path
addpath('minFunc');


%% Run L-BFGS
x = minFunc(@(B)derivativeLCL(x,y,B),B0,options);


%% Plot results