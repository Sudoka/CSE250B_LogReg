
clear

addpath('../minFunc');
addpath('../');

load('w8a.mat');

[N_train,D]=size(X0);
[N_test,D]=size(Xt0);
N=N_test+N_train;
D=D+1;
X=[X0;Xt0];
y=[Y0;Yt0];

phi_x=sparse(X)';
y=y';
phi_x(D,:)=1;


label_noise=0.1;
total_times=50;

N_valid=fix(N*0.3);
N_test=fix((N-N_valid)*0.1);
N_train=N-N_valid-N_test;

% this is the split order of the data. 
rand('twister',fix(1));
ord_all=randperm(N);

ord_valid=ord_all(1:N_valid);
phi_x_valid=phi_x(:,ord_valid);
y_valid=y(ord_valid);

ord_tt=ord_all(N_valid+1:end);
phi_x_valid=phi_x(:,ord_tt);
y_valid=y(ord_tt);

load('../gpoints.mat');
global pp;
pp=polyp;

ord_train=ord_tt;
ord_train(1:N_test)=[];
ord_test=ord_tt(1:N_test);

phi_x_train=phi_x(:,ord_train);
y_train=y(ord_train);
phi_x_test=phi_x(:,ord_test);
y_test=y(ord_test);    
    
rand('twister',fix(1));
yt_train=y_train;
ord_e=randperm(N_train);
y_train(ord_e(1:ceil(N_train*label_noise)))=yt_train(ord_e(1:ceil(N_train*label_noise)))*(-1);

for runtimes=1:total_times

    rand('twister',fix(1*runtimes));
    if runtimes==1
        theta_init=(rand(D,1)-0.5)*1e-20;  % initial theta
    else
        theta_init=(rand(D,1)-0.5);
    end

    %%%%%% This is the t-logistic regression
    
    t=1.6;    % t of t-family
    iter_xi=20;   % # of xi-step (outer loop)
    iter_theta=15; % # of theta-step (inner loop)
    lambda=2^(-1);     % regulizer=exp_t(-lambda |\theta|^2/2-g)

    [theta,xi]=tlogistic_train(phi_x_train,y_train,t,lambda,iter_xi,iter_theta, theta_init);

    err_tlog(runtimes)=tlogistic_test(theta,phi_x_train,yt_train);  % training error
    errt_tlog(runtimes)=tlogistic_test(theta,phi_x_test,y_test);    % test error

end
           
m_errt_tlog=mean(errt_tlog);
dev_errt_tlog=std(errt_tlog);