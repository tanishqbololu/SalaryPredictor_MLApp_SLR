# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:07:20 2024

@author: TANISHQ
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
 
df = pd.read_csv(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Machine Learning\2. Linear Regression\1.Simple Linear Reegression\Salary_Data.csv")
 
x = df.iloc[:,:-1] # Independent Variable
 
y = df.iloc[:,-1] # Dependent Variable

# Split data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)


# Model building

from sklearn.linear_model import LinearRegression

reg = LinearRegression() # Model

reg.fit(x_train, y_train)


# Predicting result for test
y_pred = reg.predict(x_test)


# Predict vs Actual 

comparison = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

print(comparison)


# Visulisation

plt.scatter(x_test, y_test, color='red')   # These are actual points which we kept for testing

plt.plot(x_train, reg.predict(x_train)) # This is prediction

plt.title('Salary vs Exprerience(Test set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()


# y = mx + c
# Find slope m
 
m_slope = reg.coef_
print('Slope: ',m_slope)


# Find constant c 

c_intercept = reg.intercept_
print('Constant: ',c_intercept)


# Future Predicting salaries for 20 years exp.

exp12 = m_slope*12+c_intercept
print("salary for 20 years exp: ",exp12)

# Future Predicting salaries for 70 years exp.

exp70 = m_slope*70 +c_intercept
print("salary for 70 years exp: ",exp70)

# You can write same thing without putting in formula everytime:
exp100 = reg.predict([[100]])
print("salary for 100 years exp: ",exp100)


print("salary for 10,20 ,30 ,40 years exp are: ",\
      reg.predict([[10], [20], [30], [40]]))
    

y_12 = reg.predict([[12]])
print(f"12 years of experience: ${y_12[0]:,.2f}")

# y_12[0]: Since predict returns an array, using [0] extracts the ,\
    # first (and only) value from that array.

# f before the string indicates it's an f-string, which allows for inline,\
    # variable interpolation.

#  format specifier :,.2f formats the number with commas as thousands,\
   # separators and ensures two decimal places are shown
   
 
# Statistics in code   

df.mean() 

df.median()

df['Salary'].mode()

df.var()

df.std()

# Check model performance
bias = reg.score(x_train, y_train)
variance = reg.score(x_test, y_test)


train_mse = mean_squared_error(y_train, reg.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(reg, file)
print("Model has been pickled and saved as linear_regression_model.pkl")

import os
print(os.getcwd())
