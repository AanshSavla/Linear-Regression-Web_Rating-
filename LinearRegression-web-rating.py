# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 20:15:26 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm



data = pd.read_excel("D:\AanshFolder\datasets\web-ratings-data.xlsx")
print(data.head())
plt.figure(figsize=(8,4))
plt.scatter(
        data['User'],
        data['ratings'],
        c='blue',
        )

plt.xlabel('Users')
plt.ylabel('Ratings(Out of 5)')
plt.show()

x=data['User'].values.reshape(-1,1)
print(x)
y=data['ratings'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(x,y)
print("y = {:.5}x + {:.5}".format(reg.coef_[0][0],reg.intercept_[0]))
predictions = reg.predict(x)
plt.scatter(
        data['User'],
        data['ratings'],
        c='black',
        )

plt.plot(
        data['User'],
        predictions,
        c = 'red',
        linewidth=2 
        )
plt.xlabel('Users')
plt.ylabel('Ratings(Out of 5)')
plt.show()

print("Enter user number:")
p = int(input())
pred = p*reg.coef_[0][0]+reg.intercept_[0]
print('Rating:',pred)

X = data['User']
Y = data['ratings']
X2 = sm.add_constant(X)

est = sm.OLS(y,X2)
est2 = est.fit()
print(est2.summary())



