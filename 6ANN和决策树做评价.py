#分类技术在评价中的应用（教学效果的评估）
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

df1 = pd.read_csv('C:/Users/Lenovo/Desktop/python/liyan/shuju/8.1 tae.txt')
print(df1.head())
#原数据不含列名，导致第一行数据被读成列名，应修改读取方式，自动添加列名
df = pd.read_csv('C:/Users/Lenovo/Desktop/python/liyan/shuju/8.1 tae.txt',names=['english','instructor','course','period','size','label'])
print(df.head())
df.isna().sum(axis=0)  #汇总查看每一列是否有缺失值
X=df.iloc[:,:-1]   #不包含最后一列
Y=df.iloc[:,-1]    #或写成Y=df['label']
XOH=pd.get_dummies(X,columns=['english','instructor','course','period'])  #进行one-hot编码
print(XOH.head())

#划分训练集和检验集，保证训练集、检验集与原始数据集的比例相等
TrainX,TestX,TrainY,TestY=train_test_split(XOH,Y,test_size=0.25,random_state=10,stratify=Y)
print(TrainY.value_counts(normalize=True))
print(TestY.value_counts(normalize=True))
print(Y.value_counts(normalize=True))
#建立ANN模型（输入层神经元数量即编码后属性数量为56，输出层神经元数量为3，隐藏层数量初步设定为它们的乘积再开根号）
print(np.sqrt(56*3))   #计算隐藏层数量
#Ann=MLPClassifier(hidden_layer_sizes=(13,),solver='lbfgs',activation='relu',learning_rate='constant',max_iter=3000,random_state=12)


#建立管道模型（用管道模型将标准化属性、建立模型、训练模型这些步骤串联起来，使模型依次经过每个步骤）
#Pipeline(steps)  step是各阶段构成的列表，例如[（'scl',StandardScaler）,('ann',MLPClassifier)]
pipeAnn = Pipeline([
    ('scl', preprocessing.StandardScaler()), 
    ('ann', MLPClassifier(hidden_layer_sizes=(13,), solver='lbfgs',activation='relu', learning_rate='constant', max_iter=3000, random_state=12))
])

#交叉验证
cr = cross_val_score(pipeAnn, TrainX, TrainY, scoring='accuracy',cv=10,n_jobs=-1)
print(f'初始建立的神经网络模型的平均准确率是{cr.mean():.4f},标准差为{cr.std():.4f}')
#搜索最优参数——网格搜索法 GridSearchCV(estimator,parm_grid,scoring=None,n_jobs=None,cv=None)
AnnParams={'hidden_layer_sizes':range(13,56,1),'solver':['adam'],'activation':['relu','logistic','tanh'],
            'learning_rate':['constant','invscaling']}
AnnGs=GridSearchCV(pipeAnn,param_grid=AnnParams,cv=10,n_jobs=-1)
AnnGs.fit(TrainX, TrainY)

#网格搜索的常用属性
print(AnnGs.best_params_)  #查看最优参数组合
print(AnnGs.best_score_)   #查看最优的性能指标
AnnOptimal = AnnGs.best_estimator_   #最优模型
print(AnnOptimal)          
#用最优模型重新训练模型
AnnOptimal.fit(TrainX, TrainY)
#训练集上的分类准确率
print(f'最优的神经网络模型在训练集上的准确率是{metrics.accuracy_score(TrainY,AnnOptimal.predict(TrainX)):.4f)}')
#召回率
print(f'最优的神经网络模型对于未生还的分类的召回率是{metrics.recall_score(TrainY,AnnOptimal.predict(TrainX)):.4f, pos_label=0)}')
print(f'最优的神经网络模型对于生还的分类的召回率是{metrics.recall_score(TrainY,AnnOptimal.predict(TrainX)):.4f, pos_label=1)}')
#精确率
print(f'最优的神经网络模型对于未生还的分类的精确率是{metrics.precision_score(TrainY,AnnOptimal.predict(TrainX)):.4f pos_label=0)}')
print(f'最优的神经网络模型对于生还的分类的精确率是{metrics.precision_score(TrainY,AnnOptimal.predict(TrainX)):.4f, pos_label=1)}')
#F1 score
print(f'最优的神经网络模型对于未生还的分类的f1_score是{metrics.f1_score(TrainY,AnnOptimal.predict(TrainX)):.4f, pos_label=0)}')
print(f'最优的神经网络模型对于生还的分类的f1_score是{metrics.f1_score(TrainY,AnnOptimal.predict(TrainX)):.4f, pos_label=1)}')

#检验集上的分类准确率
print(f'最优的神经网络模型在检验集上的准确率是{metrics.accuracy_score(TestY,AnnOptimal.predict(TestX)):.4f)}')
#召回率
print(f'最优的神经网络模型对于未生还的分类的召回率是{metrics.recall_score(TestY,AnnOptimal.predict(TestX)):.4f, pos_label=0)}')
print(f'最优的神经网络模型对于生还的分类的召回率是{metrics.recall_score(TestY,AnnOptimal.predict(TestX)):.4f, pos_label=1)}')
#精确率
print(f'最优的神经网络模型对于未生还的分类的精确率是{metrics.precision_score(TestY,AnnOptimal.predict(TestX)):.4f pos_label=0)}')
print(f'最优的神经网络模型对于生还的分类的精确率是{metrics.precision_score(TestY,AnnOptimal.predict(TestX)):.4f, pos_label=1)}')
#F1 score
print(f'最优的神经网络模型对于未生还的分类的f1_score是{metrics.f1_score(TestY,AnnOptimal.predict(TestX)):.4f, pos_label=0)}')
print(f'最优的神经网络模型对于生还的分类的f1_score是{metrics.f1_score(TestY,AnnOptimal.predict(TestX)):.4f, pos_label=1)}')
