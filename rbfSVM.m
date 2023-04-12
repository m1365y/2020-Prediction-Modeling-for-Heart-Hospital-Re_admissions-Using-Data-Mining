function [Err,Accuray,Sensitivity,Specificity] = rbfSVM(vec,xdata,group,K)

n = size(vec,1);
Err = zeros(n,1);
Accuray = zeros(n,1);
Sensitivity = zeros(n,1);
Specificity = zeros(n,1);
for i = 1:n
    indices = crossvalind('Kfold',group,K);

    for j = 1:K
        
        testInd = (indices == j);
        trainInd = ~testInd;
        
        SVMModel = fitcsvm(xdata(trainInd,:),group(trainInd),'Standardize',true,'KernelFunction','RBF',...
            'KernelScale','auto', 'OutlierFraction',0.05,'BoxConstraint',vec(i,2));%use vec(i,1) or 'auto'
        
        
        classout = predict(SVMModel,xdata(testInd,:));
        
        cp = classperf(group(testInd,:),classout);
        er(j) = cp.ErrorRate;
        acc(j) = cp.CorrectRate;
        Sens(j) = cp.Sensitivity;
        Spec(j) = cp.Specificity;
    end
    Err(i,1) = mean(er);
    Accuray(i,1) = mean(acc);
    Sensitivity(i,1) = mean(Sens);
    Specificity(i,1) = mean(Spec);
end

