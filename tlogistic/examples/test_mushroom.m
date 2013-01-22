
clear

addpath('../minFunc');
addpath('../');
load('mushrooms.mat');

[N,D]=size(X0);
D=D+1;
phi_x=X0';
y=Y0';
phi_x(D,:)=1;


label_noise=0.1;
total_times=10;

N_valid=fix(N*0.3);
N_test=fix((N-N_valid)*0.1);
N_train=N-N_valid-N_test;

lam_cand=[2^(-7),2^(-6),2^(-5),2^(-4),2^(-3),2^(-2),2^(-1),1,2^1,2^2,2^3,2^4,2^5,2^6,2^7];

% this is the split order of the data. 
rand('twister',fix(1));

ord_all=randperm(N);

ord_valid=ord_all(1:N_valid);
phi_x_valid=phi_x(:,ord_valid);
y_valid=y(ord_valid);

ord_tt=ord_all(N_valid+1:end);

load('../gpoints.mat');
global pp;
pp=polyp;

%%

for runtimes=1:total_times

    ord_train=ord_tt;
    ord_train(N_test*(runtimes-1)+1:N_test*runtimes)=[];
    ord_test=ord_tt(N_test*(runtimes-1)+1:N_test*runtimes);

    phi_x_train=phi_x(:,ord_train);
    y_train=y(ord_train);
    phi_x_test=phi_x(:,ord_test);
    y_test=y(ord_test);    

    rand('twister',fix(runtimes));
    yt_train=y_train;
    ord_e=randperm(N_train);
    y_train(ord_e(1:ceil(N_train*label_noise)))=yt_train(ord_e(1:ceil(N_train*label_noise)))*(-1);

    % cross validation  and Training
    
    for cv_iter=1:length(lam_cand)
        lambda=lam_cand(cv_iter);    % regulizer=exp_t(-lambda |\theta|^2/2)
            
        t=1.6;    % t of t-family
        iter_xi=20;   % # of xi-step (outer loop)
        iter_theta=15; % # of theta-step (inner loop)
        theta_init=ones(D,1)*1e-20;  % initial theta

        %%%%%% This is the t-logistic regression
        [theta,xi]=tlogistic_train(phi_x_train,y_train,t,lambda,iter_xi,iter_theta, theta_init);

        err_tlog(cv_iter,runtimes)=tlogistic_test(theta,phi_x_valid,y_valid);  % valid error
        errt_tlog(cv_iter,runtimes)=tlogistic_test(theta,phi_x_test,y_test);   % test error
    end

end

[m_err_tlog,s_lam]=min(mean(err_tlog')); % find the lambda that performs best in validation dataset

m_errt_tlog=mean(errt_tlog(s_lam,:));
dev_errt_tlog=std(errt_tlog(s_lam,:));
lambda_tlog=lam_cand(s_lam);