function [V] = SGD(m,lambda,thresh)
%SGD Minimizes using Stochastic Gradient Descent.
%
% Description:
%   This function takes in a two dimensional matrix and uses the SGD
%   algorithm to calculate the parameters related to logistic regression.
%
% Inputs:
%   m = two dimensional matrix (rows = instances, columns = features)
%   lambda = the learning rate 
%   thresh = the convergence threshold
%
% Outputs: 
%   V = vector of optimal parameters

%% Shuffles the order of the instances by randomizing the order of rows to
% go through.
m2 = zeros(size(m,1),size(m,2));
order = randperm(size(m,1));

% Put the ranomized instances form m into m2
for i =1:size(m2,1)
    m2(i,:) = m(order(i),:);
end

%% Inialization
% parameters
V = zeros(1,size(m2,2));

% initial value of partial derivative of LCL
DLCL = ones(1,size(m2,2));

% vector of the label
y = m2(:,end);

% p is the vector for logistic regression model of probability for all samples
p = zeros(size(m2,1),1);

%% The main loop to update values
% loop counter
loop =1;

while(any(DLCL>thresh))
    disp(strcat('loop = ',num2str(loop)));
    for i = 1:size(m2,1)
        p(i)= 1/(1+exp(-sum(V.*[1,m2(i,1:end-1)])));
        V = V + lambda.*(y(i)-p(i)).*[1,m2(i,1:end-1)];
    end
    
    %Check the partial derivative of parameters for convergence
    for j =1:(size(m2,2))
        DLCL(j) = sum(m2(:,j).*(y-p));
    end
    
    loop = loop +1;
end

end