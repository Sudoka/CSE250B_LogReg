% Demo of SGD

%% Demo #1
load('data/USPSN_normalized_digit0.mat');
V = SGD(data(1:1000,:), 1E-1, 1E-2);