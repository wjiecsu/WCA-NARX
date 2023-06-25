clc
clear all;
close all;

load('HBdata.mat')
trainInd=1:600;
testInd=601:size(HBdata,1);

Xtrain=HBdata(trainInd,[1,5,6,7,8,9,2,4,14]);
Xtest =HBdata(testInd,[1,5,6,7,8,9,2,4,14]);
Y=HBdata(:,18);
Ytrain  = Y(trainInd,:);
Ytest   = Y(testInd,:);

XInput1 = HBdata(trainInd,[1,5,6,7,8,9]);
XInput2 = HBdata(trainInd,[2,4,14,15]);
XtestInput1=HBdata(testInd,[1,5,6,7,8,9]);
XtestInput2=HBdata(testInd,[2,4,14,15]);

nMFs=2;        % number of MFs in each input domain
Nbs =100;      % batch size
MaxEpoch=100;  % Epoch number
alpha=.01; beta1=0.9; beta2=0.999; % AdamBounder 优化器参数
lr=0.01;
U=0.045;          %LM优化器阻尼系数参数
lambda=0.001;     %正则化参数
corr_theta=0.1;   %相关性阈值
Ratio_theta=0.1;  %总相关性阈值


% Comparsion experiment
[yPredTest0,runtime0] = TSKFNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,lr,lambda,nMFs);
[yPredTest1,runtime1] = TSK_RDA(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,MaxEpoch);
[yPredTest2,runtime2] = MBGD_RDA(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,MaxEpoch,Nbs);
[yPredTest3,runtime3] = NARX(XInput2,Ytrain,XtestInput2,lambda,lr,MaxEpoch);
[yPredTest4,runtime4] = SOFNN_ALA(Xtrain,Ytrain,Xtest,U,nMFs);
[yPredTest5,runtime5] = CWSOFNN(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,corr_theta,Ratio_theta,MaxEpoch);
[yPredTest6,runtime6] = WCA_NARX(XInput1,XInput2,Ytrain,XtestInput1,XtestInput2,nMFs,alpha,beta1,beta2,lambda,corr_theta,Ratio_theta,MaxEpoch);

%----------------------Performance-----------------------------------------------------------
runtime=[runtime0,runtime1,runtime2,runtime3,runtime4,runtime5,runtime6];
yPredTest=[yPredTest0,yPredTest1,yPredTest2,yPredTest3,yPredTest4,yPredTest5,yPredTest6];
NTest=length(yPredTest);RMSETest=[];R2Test=[];
for i=1:size(yPredTest,2)
MSE(i)=(Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest;
MAE(i)=sum(abs((Ytest-yPredTest(:,i))/NTest));
RMSETest(i)=sqrt((Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest);
R2Test(i)=1-(sum((yPredTest(:,i)-Ytest).^2)/sum((Ytest-mean(Ytest)).^2));    
end
Yall=[Ytest yPredTest];
csvwrite('./Result/AEPYall.csv',Yall);
