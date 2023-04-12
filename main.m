

clc
clear
close all

%% data loading
DataPath = 'DataSet';
dataname = 'data 300.xls';

% load dataset
data = importdata([DataPath,'/',dataname]);
data = data.data;
xdata = data(:,1:end-1);
group = data(:,end);


%% qpso-svm
D = 2;
nPop = 20;
lb = 0.01;      % lower bound for gamma(rbf param)
ub = 100;   % uper bound for gamma(rbf param)
maxit = 50;
maxeval = 1000*D;
K = 10; % number of cross fold
costFun = @(vec)rbfSVM(vec,xdata,group,K);

[xmin,fmin,histout] = QPSO(costFun,D,nPop,lb,ub,maxit,maxeval);


