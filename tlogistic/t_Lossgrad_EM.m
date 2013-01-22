%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% t_Lossgrad_EM.m   t_Lossgrad_EM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function

% t_Lossgrad_EM


% input

% theta: parameter of the model

% xi: the impact of each z_n, namely xi_n

% phi_x: feature of x, namely phi(x)

% y: label

% t: t-value

% lambdat: paramter of student't t-prior

% gt: parameter of student's t-prior


% output

% Loss: sum z_n(theta)
% Lgrad: grad (sum z_n (theta))

%%

function [Loss,Lgrad]=t_Lossgrad_EM(theta,xi,phi_x,y,t,lambdat,gt)

global pp;

D=length(theta);

thex=theta'*phi_x;
g=ppval(pp{t*10-10},thex);

tp1=exp_t(thex-g,t).^t;
tp2=exp_t(-thex-g,t).^t;

qE=(tp1-tp2)./(tp1+tp2);

Lgrad=(t-1)*lambdat*(theta.*xi(1:D))+(1-t)*phi_x*(xi(D+1:end).*(y-qE)');

Loss=[1+(1-t)*(-lambdat/2*theta'.*theta'-gt),(1+(1-t)*(y.*thex-g))]*xi;