%% Generate_data_Longservedio: generate the Long-Servedio dataset

function [phi_x,y]=Generate_data_longservedio(N)

rand('twister',fix(100));
D=21;

y=ones(1,N);
y(floor(N/2)+1:end)=-1;

phi_x=zeros(D,N);

phi_x(:,1:floor(N/8))=ones(D,1)*y(1:floor(N/8));
phi_x(:,floor(N/2)+1:floor(N/2)+1+floor(N/8))=ones(D,1)*y(floor(N/2)+1:floor(N/2)+1+floor(N/8));

phi_x(:,floor(N/8)+1:floor(N/4))=[ones(11,1);-ones(10,1)]*y(floor(N/8)+1:floor(N/4));
phi_x(:,floor(N/2)+floor(N/8)+1:floor(N/2)+floor(N/4))=[ones(11,1);-ones(10,1)]*y(floor(N/2)+floor(N/8)+1:floor(N/2)+floor(N/4));

for i=floor(N/4)+1:floor(N/2)
    N1=randperm(11);
    N2=randperm(10)+11;
    temp=ones(D,1);
    temp([N1(1:6),N2(1:4)])=temp([N1(1:6),N2(1:4)])*(-1);
    phi_x(:,i)=temp;
end

for i=floor(N/2)+floor(N/4)+1:N
    N1=randperm(11);
    N2=randperm(10)+11;
    temp=-ones(D,1);
    temp([N1(1:6),N2(1:4)])=temp([N1(1:6),N2(1:4)])*(-1);
    phi_x(:,i)=temp;
end
