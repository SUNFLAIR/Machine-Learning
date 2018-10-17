import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_df = pd.read_csv("train.csv",index_col=0)
test_df = pd.read_csv("test.csv",index_col=0)

prices = pd.DataFrame({"price":train_df['SalePrice'],'log(price+1)':np.log1p(train_df['SalePrice'])})
#可以用hist查看数据分布
#prices.hist()
#plt.show()
#这里弹出训练集的销售价格作为y值,并作平滑处理还原可以用expm1()将数据还原
y_train = np.log1p(train_df.pop('SalePrice') + 1)
#print(y_train)
#将数据剩下的部分合并起来,axis = 0 表示列
all_df = pd.concat((train_df,test_df),axis = 0)
#print(all_df.head())
#特征工程，将不方便处理或者不规整的数据给统一
#首先注意到MSSubClass的值是一个category,转化为str类型
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)
#print(all_df['MSSubClass'].value_counts())
#使用one-hot方式处理所有category数据
pd.get_dummies(all_df['MSSubClass'],prefix='MSSubClass')
#同理
all_dummy_df = pd.get_dummies(all_df)
#print(all_dummy_df.head())
#然后处理数据中的空值
#print(all_dummy_df.isnull().sum().sort_values(ascending = False).head())
#通过统计可以看到缺失的最多的LotFrontage
#我们使用平均值来填满这些空缺，也可以使用中位值等
mean_cols = all_dummy_df.mean()
#print(mean_cols.head())
#填补空缺使用函数fillna
all_dummy_df = all_dummy_df.fillna(mean_cols)
#print(all_dummy_df.isnull().sum())
#使用线性回归分类器时需要将数值特征，进行标准化
#从原始数据中找到本来就是数值类型的特征
numeric_cols = all_df.columns[all_df.dtypes != 'object']

numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std


#建立模型
#将数据分回训练和测试集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]

#print(dummy_train_df.shape,dummy_test_df.shape)

#使用岭回归 即在原始的误差平方项后加上一个正则向
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score #交叉集

#将DF格式转为np格式
X_train = dummy_train_df.values
X_test = dummy_test_df.values

#使用sklearn自带的cross_validation方法测试模型
#alphas = np.logspace(-3,2,50)#创建等比数列
#print(alphas)
#test_scores = []
# for alpha in alphas:
#     clf = Ridge(alpha)
#     test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv=10,scoring = 'neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score) )
# plt.plot(alphas,test_scores)
# plt.title("Alpha vs CV Error")
#得到学习率取值
#plt.show()

