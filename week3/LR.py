from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
iris = load_iris()
X = iris.data
Y = iris.target
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
lr=LogisticRegression(penalty='l2',solver='newton-cg',multi_class='multinomial')
lr.fit(x_train,y_train)
y_p=lr.predict(x_test)
print(lr.get_params())
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
print('____________________________')
lr=LogisticRegression(penalty='l1',solver='liblinear',multi_class='ovr')#
lr.fit(x_train,y_train)
y_p=lr.predict(x_test)
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
print('____________________________')
lr=LogisticRegression(penalty='l2',solver='liblinear',multi_class='ovr')#
lr.fit(x_train,y_train)
y_p=lr.predict(x_test)

print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))