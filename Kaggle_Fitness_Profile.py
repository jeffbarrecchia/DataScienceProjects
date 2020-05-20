#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:15:20 2020

@author: jeffbarrecchia
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model

data = pd.read_csv('~/Documents/Kaggle_Projects/CardioGoodFitness.csv')

# data.hist()
# sb.boxplot(x = 'Gender', y = 'Age', data = data)

# =============================================================================
# Creates a cross-tabulation table of product purchased vs what gender they 
# are, and also what product is purchased vs what their marital status is.
# =============================================================================

# pd.crosstab(data['Product'], data['Gender'])
# pd.crosstab(data['Product'], data['MaritalStatus'])

# =============================================================================
# Shows the overall counts of each product purchased by what gender they are.
# =============================================================================

# sb.countplot(x = 'Product', hue = 'Gender', data = data)

# =============================================================================
# Makes histograms representing Gender/Age, Gender/Income, Gender/Miles, 
# Product/Miles, and Product/Fitness
# 
# As the x axis starts at a higher point, the men that purchase our products will
# typically ride more miles than the female customers will, or at least plan to.
# 
# The products that people purchase does make an impact on how much
# they plan to ride. If I had to guess, the higher the model number, the more
# expensive/nice the product is. Logically, this typically implies that the
# people buying these products are either more inclined to use them or are higher on
# the fitness scale.
# 
# The previously pointed out observation is proven by the next histogram. It
# shows that the typical buyer of the TM195 is a 3 on the fitness scale, while 
# the typical buyer of the TM798 is a 5 on the fitness scale.
# 
# Next, the productByIncome histogram also shows that typically, higher income
# people are purchasing the TM798 over the TM195. This could mean that either
# high income correlates to high fitness, or just that if they make more money
# then why not buy the higher end product. 
# =============================================================================

# miles_hist = data.hist(by='Gender', column = 'Miles')
# milesByProduct_hist = data.hist(by='Product', column = 'Miles')
# fitnessByProduct_hist = data.hist(by='Product', column = 'Fitness')
# productByIncome_hist = data.hist(by='Product', column = 'Income')
# productByMaritalStatus_hist = data.hist(by='Product', column = 'MaritalStatus')

# =============================================================================
# Finds correlation between all variables and makes a heatmap representing 
# these values
# =============================================================================

# corr = data.corr()
# sb.heatmap(corr, annot=True)

# =============================================================================
# Sets up the regression function and stores it in variable called regr
# =============================================================================

regr = linear_model.LinearRegression()

# =============================================================================
# Regression based on their planned usage and their current fitness level to
# predict how many miles they want to ride
# =============================================================================

# x = data[['Usage', 'Fitness']]
# y = data['Miles']

# regr.fit(x, y)

# print(regr.coef_)
# print(regr.intercept_)

# print('\nThe predicted miles they want to ride based on intended usage and current fitness is', regr.intercept_, '+', '(', regr.coef_[0], '*', 'usage)', '+', '(', regr.coef_[1], '*', 'Fitness)')

# print('''\nThis does not make much sense, as if they do not intend
# to use it at all and their current fitness level is 0, how can they ride negative miles?''')

# =============================================================================
# Regression based on their Age to predict how many miles they want to ride
# =============================================================================

# x_1 = data[['Age']]
# y_1 = data['Miles']

# regr.fit(x_1, y_1)

# print(regr.coef_)
# print(regr.intercept_)

# print('\n\n\nThe predicted miles they would ride based on their Age is', regr.intercept_, '+ (', regr.coef_[0], '* their Age)')
# print('''\nThis still does not fully make sense, as if the customer 
# buying the product is 0 then their expected miles ridden is 
# still 95.32. This is allowed though as the age 0 was not 
# taken into account when making the overall regression.''')

# =============================================================================
# Regression based on current fitness level to predict how many miles they
# want to ride
# =============================================================================

# x_2 = data[['Fitness']]
# y_2 = data['Miles']

# regr.fit(x_2, y_2)

# print('\n\n\nThe predicted miles they would ride based on their current fitness level is', regr.intercept_, '+ (', regr.coef_[0], '* their current Fitness Level)')
# print('''\nThis makes sense, as the lowest fitness level you can record 
# is a 1, meaning that someone with a fitness score of 1 is 
# expected to ride is 4.98 miles.''')

# =============================================================================
# Regression based on fitness level to predict their current income
# =============================================================================

# x_3 = data[['Fitness']]
# y_3 = data['Income']

# regr.fit(x_3, y_3)

# print('\n\n\nThe predicted income of our customers based on their fitness level is', regr.intercept_, '+ (', regr.coef_, '* their Fitness Rating)')
# print('''\nThis shows a positive correlation between their fitness 
# level and their income. So as fitness goes up, income goes up.''')















