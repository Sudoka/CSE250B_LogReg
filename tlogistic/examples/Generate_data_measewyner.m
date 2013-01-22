%% Generate_data_measewyner: generate the Mease-Wyner dataset

function [phi_x,y]=Generate_data_measewyner(N)

rand('twister',fix(100));

phi_x=rand(20,N);
y=sum(phi_x(1:5,:))>=2.5;
y=y*2-1;

phi_x(21,:)=1;