function meanDLCL = derivativeLCL(x, y, B)
%DERIVATIVELCL Calculate derivative of LCL.
%
% Inputs:
%   x = features (matrix)
%   y = labels (column vector)
%   B = vector of parameters (row vector)
%
% Outputs:
%   meanDLCL = mean value of dLCL/dB
%

% get dimensions
nExamples = size(x,1); % # of training examples
nFeatures = size(x,2); % # of features

% initalize
p = zeros(nExamples,1);     % probabilities

% 1D -> 2D
BMatrix = repmat(B,nExamples,1);

% sigmoid fcn (1,nExamples)
z = sum(BMatrix.*x, 2);
p = ( 1 / ( 1 + exp(-z) ) )';

diff = y - p;
diffMatrix = repmat(diff, 1, nFeatures);
dLCL = sum(diffMatrix.*x, 1);

meanDLCL = mean( abs( dLCL ) );

end
