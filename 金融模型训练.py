#_*_coding:utf-8_*_

# 人物画像是对人的特征进行提取，例如：房产，存款，消费
# ks（打分卡、评分卡） 指标 金融 逻辑回归 评估
# 逻辑回归 做金融，svm,rf这些去验证
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LogisticRegressionCV,LinearRegression
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("./data/LoanStats3amode.csv")

Y=df['loan_status']
X=df.drop(['loan_status'],axis=1,inplace=False)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
lr = LogisticRegressionCV(multi_class='ovr',fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2', solver='lbfgs', tol=0.01)
re=lr.fit(X_train, Y_train)

# 4. 模型效果获取
r = re.score(X_train, Y_train)
print ("R值（准确率）：", r)
print ("稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))#返回的系数theta解中有没有0
print ("参数：",re.coef_)
print ("截距：",re.intercept_)#预测的概率
print(re.predict_proba(X_test))
