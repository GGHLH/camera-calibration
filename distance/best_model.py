import numpy as np
import os 
import pandas as pd
from sklearn import linear_model

## 导入回归方程
df = pd.read_csv("./distance/csv/trash.csv")
res = np.array(df)[:,1:]

## 二元非线性回归
res = np.array(res)
x = res[:,0]
y = res[:,1]
D = res[:,2]
Ind = np.array([np.ones(len(x)), x, x*x, y, y*y, x*y]).transpose(1,0)# 自变量
De  = np.array(D)   # 因变量
model = linear_model.LinearRegression()
model.fit(Ind, De) 
b1 = model.intercept_ # 常数项 b1
print(b1)
for i in range(1,6):# 依次为 b2,b3,b4,b5,b6
    print(model.coef_[i])
# D  = 25071.6534 - 2.9006 * x - 0.00049 * x*x  -114.3913 * y + 0.1338 * y*y +0.0084 *x *y

# distance = b1 + b2x + b3x^2 + b4y +b5y^2 + b6xy
# D = x^2 + y



D_pred = model.predict(Ind)
R2_score = r2_score(res[:,2], D_pred)#R^2
MSE = mean_squared_error(res[:,2], D_pred)
print("R^2_score",R2_score)
print("MSE",MSE)


coefficient = [b1] #保存求的系数，方便调用
coefficient.append(model.coef_[i]) for i in range(1,6)
columns = ["coefficient"]
df = pd.DataFrame(columns = columns, data = coef)
df.to_csv('./distance/csv/pred_coefficient.csv', encoding='utf-8')


