import numpy as np
from   matplotlib import pyplot as plt  
import seaborn as sns
import pandas  as pd
Yall=pd.read_csv('./AEPYall.csv',encoding='gbk',header=None)
Yall=Yall.values
YReal=Yall[:,0]
YPred=Yall[:,1:8]
font={'family':'Times New Roman',
      'weight':'normal',
      'size':14}
colorset =['blue','red','dusty purple','greyish','green']
markerset=['^','v','o','d','s','*']
methodname=['TSKFNN_EBP','TSK_RDA','NARX','MBGD_RDA','SOFNN_ALA','CwSOFNN','WCA_NARX']
xIteration=[i for i in range(85)]
markerIteration=[i for i in range(0,85,5)]
R2=[0.79, 0.93 ,0.87 ,0.94 ,0.86 ,0.94, 0.95]
RMSE=[0.77,0.45,0.61,0.42,0.64,0.41,0.37]
for i in range(0,7,1):
    YpredTest=YPred[:,i]
    plt.figure(figsize=[9,2],dpi=300)
    plt.plot(xIteration,YpredTest,color=sns.xkcd_rgb['red'],lw=3,label=methodname[i])
    plt.plot(xIteration,YReal,color=sns.xkcd_rgb['blue'],lw=3,label='Real Vaule')
    plt.plot(markerIteration,YpredTest[markerIteration],'^',markersize=6,markerfacecolor='white',color='red',markeredgewidth=2)
    plt.plot(markerIteration,YReal[markerIteration],'o',markersize=6,markerfacecolor='white',color='blue',markeredgewidth=2)
    plt.grid(True)
    plt.grid(True)
    plt.xticks(fontproperties='Times New Roman',size=14)
    plt.yticks(fontproperties='Times New Roman',size=14)
    plt.legend(prop=font)
    plt.xlabel(r'Samples',font)
    plt.ylabel(r'Vaule',font)
    bbox_props = dict(boxstyle="round",fc="w", ec="0.8",lw=1,alpha=0.9)
    plt.text(40,2,r"$R^2$="+str(R2[i])+" RMSE="+str(RMSE[i]),
         fontsize=12,
         fontname='Times New Roman',
         color="k",
         verticalalignment ='top', 
         horizontalalignment ='center',
         bbox =bbox_props
        )
    #plt.savefig('./AEPPrediction'+str(i)+'.png',bbox_inches = 'tight')



