% Demo of SGD

%% Demo #1
load('data/USPSN_normalized_digit0.mat');
V = SGD(data(1:100,:), 0.001, 0.01)