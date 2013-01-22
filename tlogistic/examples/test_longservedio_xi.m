clear

addpath('../minFunc');
addpath('../');

N=2000;

[phi_x,y]=Generate_data_longservedio(N);

[D,N]=size(phi_x);

label_noise=0.1;

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


        
%%%%%% This is the t-logistic regression

t=1.6;    % t of t-family
iter_xi=20;   % # of xi-step (outer loop)
iter_theta=15; % # of theta-step (inner loop)
lambda=2^(-1);     % regulizer=exp_t(-lambda |\theta|^2/2-g)
theta_init=(rand(D,1)-0.5);

[theta,xi]=tlogistic_train(phi_x_train,y_train,t,lambda,iter_xi,iter_theta, theta_init);


           
%%%%%% plot the \xi plot
interval=80;

figure
hold on
box on
x=([(min(xi)):((max(xi))-(min(xi)))/interval:(max(xi))]);
BinH(1,:)=(hist(xi(ord_e(1:ceil(N_train*label_noise))),(x)));
BinH(2,:)=(hist(xi(ord_e(ceil(N_train*label_noise)+1:end)),(x)));
bar((x),BinH','stacked')

hold off