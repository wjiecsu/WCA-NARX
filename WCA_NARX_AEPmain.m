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
MaxEpoch=100;  % Epoch number
alpha=.01; beta1=0.9; beta2=0.999; % AdamBounder 优化器参数
U=0.045;          %LM优化器阻尼系数参数
lambda=0.001;     %正则化参数
corr_theta=0.1;   %相关性阈值
Ratio_theta=0.1;  %总相关性阈值

%% 主程序 [yPredTest6,runtime6] = WCA_NARX(XInput1,XInput2,Ytrain,XtestInput1,XtestInput2,nMFs,alpha,beta1,beta2,lambda,corr_theta,Ratio_theta,MaxEpoch);
tic;
[XInput1,XInput2,Ytrain,XtestInput1,XtestInput2,outputps] =NormilzeData(XInput1,XInput2,Ytrain,XtestInput1,XtestInput2);
rand('seed',3); %随机种子固定
[N,M1]=size(XInput1);
[~,M2]=size(XInput2);
[NTest,~]=size(XtestInput1);
% Defination
nMFsVec=nMFs*ones(M1,1);
nMFsInit=nMFs;
nRules=nMFs^M1; % number of rules
C=zeros(M1,nMFs);
Sigma=C;
W=zeros(nRules,M2+1);
idMFsList=[];
for r=1:nRules
    idsMFs=idx2vec(r,nMFsVec); %idx2vec函数
    idMFsList=[idMFsList;idsMFs];
end

% Initialization
for m=1:M1
    C(m,:)=linspace(min(XInput1(:,m))+0.1,max(XInput1(:,m))-0.1,nMFs);
    Sigma(m,:)=std(XInput1(:,m));
end
minSigma=min(Sigma(:));
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0;
shuffle=randperm(N);
Epoch=1;iter=1;
eps=1e-8;
Ruleset=[];
% Iteration
while Epoch<=MaxEpoch
    % 初始化为空
    mu=zeros(M1,nMFs);
    deltaC=zeros(M1,nMFs); deltaSigma=deltaC;  deltaW=lambda*W; deltaW(:,1)=0; % 每个Epoch都会置为0
    f=ones(N,nRules); % firing level of rules
    disp(['[ARFNNX:] ','complete Epoch ->',num2str(Epoch),' steps'])
    
    fBarset=[];
    % 前向传播
    for i=1:N
        for m=1:M1 % membership grades of MFs
            mu(m,:)=exp(-(XInput1(i,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        for r=1:nRules
            idsMFs=idMFsList(r,:);
            for m=1:M1
                f(i,r)=f(i,r)*mu(m,idsMFs(m));
            end
        end
        fBar=f(i,:)/(sum(f(i,:))+eps);
        fBarset=[fBarset;fBar];
        yR=[1 XInput2(i,:)]*W';                      %Wx+b W:R*(m+1)
        ypred(i)=fBar*yR';                          %预测
        

        
        % 计算梯度
        for r=1:nRules
            temp=(ypred(i)-Ytrain(i))*(yR(r)*sum(f(i,:))-f(i,:)*yR')/sum(f(i,:))^2*f(i,r);
            if ~isnan(temp) && abs(temp)<inf
                vec=idMFsList(r,:);
                %% delta of c, sigma, W, and b
                for m=1:M1
                    deltaC(m,vec(m))=deltaC(m,vec(m))+temp*(XInput1(i,m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=deltaSigma(m,vec(m))+temp*(XInput1(i,m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    
                end
                for m=1:M2
                    deltaW(r,m+1)=deltaW(r,m+1)+(ypred(i)-Ytrain(i))*fBar(r)*XInput2(i,m);
                end
                %% delta of b0
                deltaW(r,1)=deltaW(r,1)+(ypred(i)-Ytrain(i))*fBar(r);
            end
        end
    end
    
    % AdaBound
    lb=alpha*(1-1/((1-beta2)*iter+1));
    ub=alpha*(1+1/((1-beta2)*iter));
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^iter);
    vCHat=vC/(1-beta2^iter);
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat; %更新C
    
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^iter);
    vSigmaHat=vSigma/(1-beta2^iter);
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma  =max(minSigma,Sigma-lrSigma.*mSigmaHat);  %更新Sigma
    
    mW=beta1*mW+(1-beta1)*deltaW;
    vW=beta2*vW+(1-beta2)*deltaW.^2;
    mWHat=mW/(1-beta1^iter);
    vWHat=vW/(1-beta2^iter);
    lrW=min(ub,max(lb,alpha./(sqrt(vWHat)+10^(-8))));
    W=W-lrW.*mWHat;
    
    %%%% 自组织网络
    pearson_corr=[];%开始 进行增长和剪枝策略
    for u=1:nRules
        pearson_corr(u)=corr(fBarset(:,u),Ytrain);%fBarset:N*R
    end
    pearson_sum(Epoch)=sum(abs(pearson_corr));
    
    %-------------------------------增长-----------------------------------
    if (pearson_sum(Epoch)<max(pearson_sum(1:Epoch-1)))&(Epoch~=MaxEpoch)
        fprintf(2,['start the grow operation\n'])
        [~,max_index]=max(abs(pearson_corr));
        nMFs=nMFs+1;                   % 增长,只对最大的值进行增长
        nRules=nRules+1;
        idsMFs=nMFs*ones(1,M1);         % 新增的
        idMFsList=[idMFsList;idsMFs];
        max_idMFs=idMFsList(max_index,:);
        for k=1:M1 %就是对相关性最大的隐节点分裂
            if max_idMFs(k)  <=1
                newC(k,1)    =1/2*C(k,max_idMFs(k))+C(k,max_idMFs(k)); 
                newSigma(k,1)=1/2*Sigma(k,max_idMFs(k))+Sigma(k,max_idMFs(k));
            else
                newC(k,1)    =1/2*C(k,max_idMFs(k)-1)+1/2*C(k,max_idMFs(k)); 
                newSigma(k,1)=1/2*Sigma(k,max_idMFs(k)-1)+1/2*C(k,max_idMFs(k));
            end    
        end
        %更新参数
        C=[C,newC];
        Sigma=[Sigma,newSigma];
        Wadd=W(max_index,:);
        W=[W;Wadd];
        
        %更新梯度
        mC=[mC,zeros(M1,1)];
        vC=[vC,zeros(M1,1)];
        mSigma=[mSigma,zeros(M1,1)];
        vSigma=[vSigma,zeros(M1,1)];
        mW=[mW;zeros(1,M2+1)];
        vW=[vW;zeros(1,M2+1)];
    end
    
    %----------------------------剪枝策略-----------------------------------
    Ratio=sum(abs(pearson_corr(find(abs(pearson_corr)<corr_theta))))/pearson_sum(Epoch);
    if (Ratio<Ratio_theta)&(Ratio~=0)&(Epoch~=MaxEpoch)% 删减，对最小的值进行删除
        fprintf(2,['start the delete operation\n'])
        min_index=find(abs(pearson_corr)<corr_theta);
        Rules_index=1:nRules;
        Is_detelte=ismember(Rules_index,min_index);
        new_Rules_index=Rules_index(~Is_detelte);
        idMFsList=idMFsList(new_Rules_index,:);
        nRules=nRules-size(min_index,2);
        
        %更新参数
        W=W(new_Rules_index,:);

        
        %更新梯度
        mW=mW(new_Rules_index,:);
        vW=vW(new_Rules_index,:);
    end
    %%% Record
    Ruleset=[Ruleset;nRules];
    RMSE(Epoch,1)=sqrt(sum((ypred'-Ytrain).^2)/N);
    %迭代参数更新
    iter=iter+1;
    Epoch=Epoch+1;
end
runtime=toc;
%%%%%% Test yPred
tic
fTest=ones(NTest,nRules); % firing level of rules
for i=1:NTest
    for m=1:M1 % membership grades of MFs
        mu(m,:)=exp(-(XtestInput1(i,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
    end
    
    for r=1:nRules % firing levels of rules
        idsMFs=idMFsList(r,:);
        for m=1:M1
            fTest(i,r)=fTest(i,r)*mu(m,idsMFs(m));
        end
    end
end
yR=[ones(NTest,1) XtestInput2]*W';
yPredTest=sum(fTest.*yR,2)./sum(fTest,2); % prediction
yPredTest =InverseNormilzeData(yPredTest,outputps);
Testtime=toc;

%%评价指标
MSE=(Ytest-yPredTest)'*(Ytest-yPredTest)/NTest;
MAE=sum(abs((Ytest-yPredTest)/NTest));
RMSETest=sqrt((Ytest-yPredTest)'*(Ytest-yPredTest)/NTest);
R2Test=1-(sum((yPredTest-Ytest).^2)/sum((Ytest-mean(Ytest)).^2));    

function vec=idx2vec(idx,nMFs)
% Convert from a scalar index of the rule to a vector index of MFs
vec=zeros(1,length(nMFs));
prods=[1; cumprod(nMFs(end:-1:1))];
if idx>prods(end)
    error('Error: idx is larger than the number of rules.');
end
prev=0;
for i=1:length(nMFs)
    vec(i)=floor((idx-1-prev)/prods(end-i))+1;
    prev=prev+(vec(i)-1)*prods(end-i);
end
end

function [XInput1,XInput2,Ytrain,XtestInput1,XtestInput2,outputps] =NormilzeData(XInput1,XInput2,Ytrain,XtestInput1,XtestInput2)
XInput1=XInput1';
XInput2=XInput2';
Ytrain=Ytrain';
XtestInput1=XtestInput1';
XtestInput2=XtestInput2';
[XInput1,inputps1] = mapminmax(XInput1,0,1);
[XInput2,inputps2] = mapminmax(XInput2,0,1);
[Ytrain,outputps]= mapminmax(Ytrain,0,1);    %归一化后的数据
XtestInput1=mapminmax('apply',XtestInput1,inputps1); %测试集数据归一化
XtestInput2=mapminmax('apply',XtestInput2,inputps2); %测试集数据归一化
% 还原
XInput1=XInput1';
XInput2=XInput2';
Ytrain=Ytrain';
XtestInput1=XtestInput1';
XtestInput2=XtestInput2';
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%反归一化
end
