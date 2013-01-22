%% This function takes in a two dimensional matrix and uses the SGD
%% algorithm to calculate the parameters related to logistic regression
%% Inputs: m = two dimensional matrix. Rows are instances and columns are
%%             features
%%         lambda = the learning rate
%%         mu = regularization constant
%%         thresh = the convergence threshold
%% Outputs: V = vector of parameters
%%          T = test error(expected value of the loss function)
%%          Var = variance of test error
function [V,T,Var] = SGD(m,lambda,mu,thresh)
    % Shuffles the order of the instances by randomizing the order of rows to
    % go through.
    m2 = zeros(size(m,1),size(m,2));
    order = randperm(size(m,1));
    % Put the ranomized instances form m into m2
    for i =1:size(m2,1)
        m2(i,:) = m(order(i),:);
    end
    % V is the vector of parameters.  
    V = zeros(1,size(m2,2));
    % Calcualte the initial value of partial derivative of LCL.
    DLCL = ones(1,size(m2,2));
    % y is the vector of label
    y = m2(:,end);
    % p is the vector for logistic regression model of probability for all
    % samples
    % Initialize p 
    p = zeros(size(m2,1),1);
    % The main loop to update values
    % loop counter
    loop =1;
    % BetaDiff is the change in the objective function. This has to be less
    % then thresh
    BetaDiff =1;
    % Beta is the objective function
    Beta =0;
    % Go through the loops to minimize the objective function
    while(BetaDiff >= thresh)
        disp(strcat('loop = ',num2str(loop)));
        for i = 1:size(m2,1)
            % p(i) is the probability that y =1 given x(i)
            p(i)= 1/(1+exp(-sum(V.*[1,m2(i,1:end-1)])));
            % Update the parameters with regularization
            V = V + lambda.*((y(i)-p(i)).*[1,m2(i,1:end-1)]-2*mu.*V);
        end
        %Check the change in objective function for convergence
        newBeta = sum(-log((p.^y).*(1-p).^(1-y))) + mu.*sum(V.^2); 
        BetaDiff = abs(newBeta-Beta);
        disp(strcat('BetaDiff = ',num2str(BetaDiff)));
        loop = loop +1;
        % Update the value of the objective function
        Beta = newBeta;
        disp(strcat('Beta = ',num2str(Beta)));
    end
    % After the objective function converges, calculate test error and the
    % variance of test error
    % Test error is the expected value of the loss function which is the
    % average of -log(p)
    T = sum(-log((p.^y).*(1-p).^(1-y)))./size(y,1);
    Var = sum((-log((p.^y).*(1-p).^(1-y))-T).^2)./size(y,1);
end

