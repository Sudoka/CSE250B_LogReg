%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% t_Loss_EM.m   t_Loss_EM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function

% t_Loss_EM


% input

% theta: parameter of the model

% phi_x: feature of x, namely phi(x)

% y: label

% t: t-value

% lambdat: paramter of student't t-prior

% gt: parameter of student's t-prior


% output

% L: z_n(theta)


%%

function L=t_Loss_EM(theta,phi_x,y,t,lambdat,gt)

global pp;

[D,N]=size(phi_x);

thex=theta'*phi_x;
g=ppval(pp{t*10-10},thex);

L=zeros(1,N+D);

L(D+1:end)=1+(1-t)*(y.*thex-g);
L(1:D)=1+(1-t)*(-lambdat/2*theta'.*theta'-gt);

L=L';
