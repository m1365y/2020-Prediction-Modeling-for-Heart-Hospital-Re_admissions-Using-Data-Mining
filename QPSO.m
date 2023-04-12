function [xmin,fmin,histout] = QPSO(fun,D,nPop,lb,ub,maxit,maxeval)

% INPUT:
%   fun     : function handle for optimization
%   D       : problem dimension (number of variables)
%   nPop    : number of particles in the swarm
%   lb      : lower bound constrain
%   ub      : upper bound constrain
%   maxit   : max number of iterations
%   maxeval : max number of function evaluations

% OUTPUT:
%   xmin    : best solution found
%   fmin    : function value at the best solution, f(xmin)
%   histout : record of function evaluations and fitness value by iteration

% QPSO parameters:
w1 = 0.5;
w2 = 1.0;

c1 = 1.5;
c2 = 1.5;

% Initializing solution
x = unifrnd(lb,ub,[nPop,D]);

% Evaluate initial population
pbest = x;

histout = zeros(maxit,2);

[f_x,Acc_x,Sens_x,Spec_x] = fun(x);

fval = nPop;

f_pbest = f_x;

[~,g] = min(f_pbest);
gbest = pbest(g,:);
f_gbest = f_pbest(g);

Acc_gbest = Acc_x(g); Sens_gbest = Sens_x(g); Spec_gbest = Spec_x(g);

it = 1;

histout(it,1) = fval;
histout(it,2) = f_gbest;

while it < maxit && fval < maxeval
    formatSpec = 'Iteration = %1d, Err = %2.3f, Acc = %2.3f, Sensitivity = %2.3f, Specificity = %2.3f \n';
    fprintf(formatSpec,it,f_gbest,Acc_gbest,Sens_gbest,Spec_gbest)
    
    alpha = (w2 - w1) * (maxit - it)/maxit + w1;
    mbest = sum(pbest)/nPop;
    
    for i = 1:nPop
        
        fi = rand(1,D);
        
        p = (c1*fi.*pbest(i,:) + c2*(1-fi).*gbest)/(c1 + c2);
        
        u = rand(1,D);
        
        b = alpha*abs(x(i,:) - mbest);
        v = log(1./u);
        
        if rand < 0.5
            x(i,:) = p + b .* v;
        else
            x(i,:) = p - b .* v;
        end
        
        % Keeping bounds
        x(i,:) = max(x(i,:),lb);
        x(i,:) = min(x(i,:),ub);
        
        [f_x(i),Acc_x(i),Sens_x(i),Spec_x(i)] = fun(x(i,:));
        
        fval = fval + 1;
        
        if f_x(i) < f_pbest(i)
            pbest(i,:) = x(i,:);
            f_pbest(i) = f_x(i);
            
        end
        
        if f_pbest(i) < f_gbest
            gbest = pbest(i,:);
            f_gbest = f_pbest(i);
            
            Acc_gbest = Acc_x(i);
            Sens_gbest = Sens_x(i);
            Spec_gbest = Spec_x(i);
            
        end
        
    end
    
    it = it + 1;
    
    histout(it,1) = fval;
    histout(it,2) = f_gbest;
    
end

xmin = gbest;
fmin = f_gbest;

histout(it+1:end,:) = [];

h = figure;
% semilogy(histout(:,1),histout(:,2))
plot(histout(:,1),histout(:,2))
title('Q-PSO')
xlabel('Function evaluations')
ylabel('Classification Error')
grid on
saveas(h,'RESULTS\figure','jpeg');
saveas(h,'RESULTS\figure','fig');
save('RESULTS\result','f_gbest','Acc_gbest','Sens_gbest','Spec_gbest');
end