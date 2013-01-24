%% MTest - matrix to train with
%% MValid - matrix to evaluate T-error
%% mu - regularization constant

function [V,T,Var] = LBFGS(MTest, MValid, mu)
	
	YTest = MTest(:,end);
	function [F,G] = objective(x)
		P = zeros(size(MTest,1),1);
		GradientSum = zeros(1,size(MTest,2));
		ObjSum = 0;
		%% MinFunc deals with column vectors, not row vectors
		Vcur = x';
		for i=1:size(MTest,1)
		    P(i) = 1/(1+exp(-sum(Vcur.*[1,MTest(i,1:end-1)])));
			GradientSum = GradientSum - (YTest(i) - P(i)).*[1,MTest(i,1:end-1)];
			if YTest(i) > 0.5
			    ObjSum = ObjSum - log(P(i));
			else
			    ObjSum = ObjSum - log(1 - P(i));
			end
		end
		%% compute LCL
		F = ObjSum + mu*norm(Vcur);
		%fprintf(1,'F = %.40e\n',F);
		%% compute derivative of LCL (gradient)
		G = (GradientSum - (2*mu.*Vcur))';
    end
	
	options = [];
	options.Method = 'lbfgs';
	%options.DerivativeCheck = 'on';
	Vstart = zeros(1,size(MTest,2));
	Vstart(1) = 1;
	
	%r = checkgrad2(@objective,Vstart');
	%disp(strcat('r = ',num2str(r)));
	
	%% Use minFunc to converge
	[Vend,f,exitflag,output] = minFunc(@objective,Vstart',options);
	V = Vend';
	
	yValid = MValid(:,end);
	pValid = zeros(size(MValid,1),1);
	for i=1:size(MValid,1)
        pValid(i) = 1/(1+exp(-sum(V.*[1,MValid(i,1:end-1)])));
    end
	
	T = sum(-log((pValid.^yValid).*(1-pValid).^(1-yValid)))./size(yValid,1);
    Var = sum((-log((pValid.^yValid).*(1-pValid).^(1-yValid))-T).^2)./size(yValid,1);
end