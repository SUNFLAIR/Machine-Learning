import numpy as np

x=np.arange(6).reshape(3,2)
from sklearn.preprocessing import PolynomialFeatures
s=PolynomialFeatures(2)
k=s.fit_transform(x)
print(k)