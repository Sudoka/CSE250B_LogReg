%% Load data
clear all;
filename = 'Web_normalized.csv';
savename = strcat(filename(1:end-4),'.mat');

% load data
data = load(filename);
fprintf('Loaded: %s \n', filename)


% save as .mat
save(savename, 'data')
fprintf('Saved: %s \n', savename)