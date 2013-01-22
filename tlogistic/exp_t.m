%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% exp_t.m   exp_t

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function

% exp_t:  compute exp_t(x)


% input

% x: a vector of Nx1
% t: t-value

% output

% y: a vector of the same size as x, with y=exp_t(x)


%%

function y=exp_t(x,t)

y=max(1e-60,(1+(1-t)*x)).^(1/(1-t));