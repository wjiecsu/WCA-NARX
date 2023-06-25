function [yPredTest,runtime]=NARX(Xtrain,Ytrain,Xtest,lambda,lr,MaxEpoch)
input_train  = Xtrain'  ;
output_train = Ytrain'  ;
input_test   = Xtest'   ;
rand('seed',4); %随机种子固定

%样本输入输出数据归一化
[inputn,inputps]  =mapminmax(input_train,0,1);
[outputn,outputps]=mapminmax(output_train,0,1);

% 节点个数
hiddennum=6;
% 构建网络
% net=newff(P,T,[S1 S2 S(N-1)],{TF1,TF2,..,TFN},BTF,BLF,PF,IPF,OPF,DDF);
% P输入
% T输出
% [S1 S2 S(N-1)]隐含层
% {TF1,TF2,..,TFN}传递函数
% BTF 优化函数
% BLF 梯度学习方法 'learngdm'
% PF  损失函数 'mse'
% IPF 输入处理函数 ”fixunknowns”、“removeconstantrows”、“mapinmax”;
% OPT 输出处理函数 “removeconstantrows”或“mapminmax”；
% DDF 输入数据划分 'dividerand'
% newcf多层神经网络 newff单层神经网络 级联前向BP网络 newfftd 函数用来创建一个存在输入延迟的前向型网络
net=newff(inputn,outputn,hiddennum);
net.performFcn='mse';
net.performParam.regularization=lambda;
net.trainFcn = 'trainlm';
net.layers{1}.transferFCn='tansig' ;%设置隐藏层'tansig'...
net.layers{2}.transferFCn='purelin';%设置输出层'logsig'...
% 网络优化函数
% 梯度下降法                  traingd
% 有动量的梯度下降法          traingdm
% 自适应lr梯度下降法          traingda
% 自适应lr动量梯度下降法      traingdx
% 弹性梯度下降法              trainrp
% Fletcher-Reeves共轭梯度法   traincgf
% Ploak-Ribiere共轭梯度法     traincgp
% Powell-Beale共轭梯度法      traincgb
% 量化共轭梯度法              trainscg
% 拟牛顿算法                  trainbfg
% 一步正割算法                trainoss
% Levenberg-Marquardt        trainlm
net.trainParam.epochs = MaxEpoch;   %`训练次数
net.trainParam.lr=lr;               %`学习速率（权重阈值的调整幅度）
net.trainParam.goal=1e-6;           %`误差精度
%trainlm函数使用的是Levenberg-Marquardt训练函数，Mu是算法里面的一个参数：

%训练网络
[net,per2]=train(net,inputn,outputn);
% BP网络预测
input_test=mapminmax('apply',input_test,inputps);%测试集数据归一化
prediction_test=sim(net,input_test);             %预测值
yc=mapminmax('reverse',prediction_test,outputps);%反归一化
yPredTest=yc';
runtime=toc;
end


