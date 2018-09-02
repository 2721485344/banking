#_*_coding:utf-8_*_
import numpy as np
import pandas as pd
import sys
# 1000行1兆
df=pd.read_csv("./data/LoanStats3a.csv/LoanStats3a.csv",skiprows=1,low_memory=True)
# skiprows=1 从第几行读取 low_memory=True 分数据类型存储
df.drop(['id','member_id','emp_title','sub_grade'],axis=1,inplace=True)  #inplace=True 是否替换原表
# df.drop('member_id',axis=1,inplace=True)
df.term.replace(to_replace='[^0-9]+',value='', inplace=True, regex=True)
# print(df.term.value_counts()) 查看分类，统计
df.int_rate.replace(to_replace='%',value='', inplace=True, regex=True)
# df.drop(['emp_title','sub_grade'],1,inplace=True) #删除列
df.emp_length.replace(to_replace='n/a',value=np.nan,inplace=True)
df.emp_length.replace(to_replace='[^0-9]+',value='', inplace=True, regex=True)
df.dropna(axis=(0,1),how='all',inplace=True)#删除空行和列
# df.dropna(axis=1,how='all',inplace=True)#删除空列
# df.dropna(axis=0,how='all',inplace=True)#删除空行
# 数据太少，删除没有用的列
df.drop(['next_pymnt_d','mths_since_last_record','desc','debt_settlement_flag_date','settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term'],axis=1,inplace=True)
# print(df.info())
# 删除float重复值较多的特征 (统计float64类型不同值的参数个数)
# for col in df.select_dtypes(include=['float64']).columns:
    # print('col {} has {}'.format(col,len(df[col].unique())))
df.drop(['tax_liens','pub_rec_bankruptcies','delinq_amnt','chargeoff_within_12_mths','acc_now_delinq','policy_code','collections_12_mths_ex_med','out_prncp_inv','out_prncp','total_acc','pub_rec','open_acc','mths_since_last_delinq','inq_last_6mths','delinq_2yrs'],axis=1,inplace=True)
# print(df.info())


df.drop(['title','term','grade','emp_length','home_ownership','verification_status','issue_d','pymnt_plan','purpose','zip_code','addr_state','earliest_cr_line','initial_list_status','last_pymnt_d','last_credit_pull_d','application_type','hardship_flag','disbursement_method','debt_settlement_flag'],axis=1,inplace=True)
# for col in df.select_dtypes(include=['object']).columns:
#     print('col {} has {}'.format(col,len(df[col].unique())))
# print(df.loan_status.value_counts())
# 对标签进行处理
df.loan_status.replace('Fully Paid',int(1),inplace=True)
df.loan_status.replace('Charged Off',int(0),inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid',np.nan,inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off',np.nan,inplace=True)
# print(df.info())
# print(df.loan_status.value_counts())
# 删除标签是nan的实例
df.dropna(subset=['loan_status'],axis=0, how='any',inplace=True)
# 用0去填充所有的空值
df.fillna(0,inplace=True)
df.fillna(0.0,inplace=True)
# 删除相关性较强的列（特征）相关系数
df.drop(['total_pymnt','funded_amnt','loan_amnt'],1,inplace=True)
cor=df.corr()
cor.iloc[:,:]=np.tril(cor,k=-1)#矩阵，上三角
cor=cor.stack()#相关系数 重构
cor=cor[(cor>0.55)|(cor<-0.55)]  #删除大于0.95 贡献性的特征删掉
# print(df.info())
# 如果有object 或者 不是float 类型的数据 做哑变量（带有百分号的）
df=pd.get_dummies(df) #离散特征的编码分为两种情况
df.to_csv("./data/LoanStats3amode.csv")#保存经过哑变量处理过的数据
print(df.info())