
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv('train.csv')
# print(data_train.head())
# print(data_train.info())
# print(data_train.describe()
# print(data_train['SalePrice'].describe())
# plt.figure(figsize=(10, 6))
sns.set_style("darkgrid")

# sns.distplot(data_train['SalePrice'])
# fig=plt.figure()
# res=stats.probplot(data_train['SalePrice'],plot=plt)
plt.show()
print("Skewness: %f" % data_train['SalePrice'].skew())
print("Kurtosis: %f" % data_train['SalePrice'].kurt())
sns.distplot(np.log(data_train['SalePrice']))
# plt.show()

# varName = "LotArea"
# data_train.plot.scatter(x=varName,y="SalePrice",ylim=(0,800000))
# plt.show()

# b.类别特征分析

# varName="CentralAir"
# fig=sns.barplot(x=varName,y="SalePrice",data=data_train)
# fig.axis(ymin=0,ymax=800000)
# plt.show()

# varName = "OverallQual"
# fig = sns.boxplot(x=varName,y="SalePrice",data=data_train)
# fig.axis(ymin=0,ymax=800000)
# plt.show()


# var = 'Neighborhood'
# fig = sns.boxplot(x=var, y="SalePrice", data=data_train)
# fig.axis(ymin=0, ymax=800000)
# plt.show()

# #更加科学的分析
# corrmat=data_train.corr()
# sns.heatmap(corrmat,vmax=0.8,square=True,cmap='YlGnBu')
# plt.show()

from sklearn import preprocessing
# f_names = ['CentralAir','Neighborhood']
# for i in f_names:
#     label=preprocessing.LabelEncoder()
#     data_train[i]=label.fit_transform(data_train[i])
# corrmat=data_train.corr()
# sns.heatmap(corrmat,vmax=0.8,square=True,cmap='YlGnBu')
# plt.show()

# 开始模拟数据
from sklearn import  preprocessing
from sklearn import linear_model,svm,gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# cols = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','TotRmsAbvGrd','YearBuilt']
# x = data_train[cols].values
# y = data_train['SalePrice'].values
# x_scaled=preprocessing.StandardScaler().fit_transform(x)
# y_scaled=preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
# X_train,X_test,y_train,y_test=train_test_split(x_scaled,y_scaled,test_size=0.33,random_state=42)


# clfs={
#     'svm':svm.SVR(),
#     'RandomForestRegressor':RandomForestRegressor(n_estimators=400),
#     'BayesinaRidge':linear_model.BayesianRidge()
#
# }
# for clf in clfs:
#     try:
#         clfs[clf].fit(X_train,y_train)
#         y_pred=clfs[clf].predict(X_test)
#         print(clf+'cost:'+str(np.sum(y_pred-y_test)/len(y_pred)))
#     except Exception as e:
#         print(clf+"error")
# 由上面结果选择随机森林回归算法，为了更直观地观察训练结果，我将显示一下未归一化数据的预测效果。
cols = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','TotRmsAbvGrd','YearBuilt']
x = data_train[cols].values
y = data_train['SalePrice'].values
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)
clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print(y_pred)
print(sum((abs(y_pred - y_test))/len(y_pred)))

data_test=pd.read_csv('test.csv')
#
# print(data_test['GarageCars'].describe())
# data_test[ ['GarageCars'] ].fillna(1.766118)
# data_test[ ['TotalBsmtSF']].fillna(1046.117970)
# data_test[cols].fillna(data_test[cols].mean())
# print(data_test[cols].isnull().sum())
cols2 = ['OverallQual','GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
cars = data_test['GarageCars'].fillna(1.766118)
bsmt = data_test['TotalBsmtSF'].fillna(1046.117970)
data_test_x = pd.concat( [data_test[cols2], cars, bsmt] ,axis=1)
print(data_test_x.isnull().sum())
