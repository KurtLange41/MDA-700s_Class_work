#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LinearRegression
import seaborn as sns


# In[2]:


#Importing Data
df2015 = pd.read_csv("C:/Users/kurtl/OneDrive/Documents\MDA-821 Capstone/MLB Stats 2015.csv")
df2016 = pd.read_csv("C:/Users/kurtl/OneDrive/Documents\MDA-821 Capstone/MLB Stats 2016.csv")
df2017 = pd.read_csv("C:/Users/kurtl/OneDrive/Documents\MDA-821 Capstone/MLB Stats 2017.csv")
df2018 = pd.read_csv("C:/Users/kurtl/OneDrive/Documents\MDA-821 Capstone/MLB Stats 2018.csv")
df2019 = pd.read_csv("C:/Users/kurtl/OneDrive/Documents\MDA-821 Capstone/MLB Stats 2019.csv")
df2021 = pd.read_csv("C:/Users/kurtl/OneDrive/Documents\MDA-821 Capstone/MLB Stats 2021.csv")

#Citations
#https://medium.com/machine-learning-with-python/linear-regression-implementation-in-python-2de514d3a34e
#https://www.statista.com/statistics/193466/total-league-revenue-of-the-mlb-since-2005/#:~:text=In%202021%2C%20the%20combined%20revenue,at%20318.53%20million%20U.S.%20dollars.


# In[3]:


df2015


# In[4]:


# 2015 MLB DATA
df2015 = df2015.drop(df2015.columns[[0, 1, 2, 3, 10]], axis=1) 
df2015 = df2015[["b_home_run","b_k_percent","b_bb_percent","batting_avg","b_swinging_strike","exit_velocity_avg"]].mean()
print(df2015)


# In[5]:


df2016


# In[6]:


# 2016 MLB DATA
df2016 = df2016.drop(df2016.columns[[0, 1, 2, 3, 10]], axis=1) 
df2016 = df2016[["b_home_run","b_k_percent","b_bb_percent","batting_avg","b_swinging_strike","exit_velocity_avg"]].mean()
print(df2016)


# In[7]:


df2017


# In[8]:


# 2017 MLB DATA
df2017 = df2017.drop(df2017.columns[[0, 1, 2, 3, 10]], axis=1) 
df2017 = df2017[["b_home_run","b_k_percent","b_bb_percent","batting_avg","b_swinging_strike","exit_velocity_avg"]].mean()
print(df2017)


# In[9]:


df2018


# In[10]:


# 2018 MLB DATA
df2018 = df2018.drop(df2018.columns[[0, 1, 2, 3, 10]], axis=1) 
df2018 = df2018[["b_home_run","b_k_percent","b_bb_percent","batting_avg","b_swinging_strike","exit_velocity_avg"]].mean()
print(df2018)


# In[11]:


df2019


# In[12]:


# 2019 MLB DATA
df2019 = df2019.drop(df2019.columns[[0, 1, 2, 3, 10]], axis=1) 
df2019 = df2019[["b_home_run","b_k_percent","b_bb_percent","batting_avg","b_swinging_strike","exit_velocity_avg"]].mean()
print(df2019)


# In[13]:


df2021


# In[14]:


# 2021 MLB DATA
df2021 = df2021.drop(df2021.columns[[0, 1, 2, 3, 10]], axis=1) 
df2021 = df2021[["b_home_run","b_k_percent","b_bb_percent","batting_avg","b_swinging_strike","exit_velocity_avg"]].mean()
print(df2021)


# In[15]:


# Merging All DF

df = pd.concat((df2015, df2016, df2017, df2018, df2019, df2021), axis = 1)

df


# In[16]:


#Renaming all columns

df.columns = ['2015', '2016', '2017', '2018', '2019', '2021']

df


# In[17]:


#Adding MLB revenue in Billion

df2 = {'2015': 8.39, '2016': 9.03, '2017': 9.46, '2018': 9.90, '2019': 10.37, '2021': 9.56}
df = df.append(df2, ignore_index = True)
  
display(df)


# In[18]:


#Renaming Rows

df.rename(index = {0: "Home_Run_Avg",
                     1:"Avg_K",
                      2:"Avg_Walk",
                      3:"Avg_BA",
                      4:"Avg_SM",
                      5:"Avg_ExitVelo",
                      6:"Revenue_in_B"},
                                 inplace = True)

df


# In[19]:


#Transposing the Dataframe
df1 = df.T
df1


# In[20]:


#Looking for Highest Correlation to Revenue
df1.corr()


# In[21]:


#Plotting Heatmap of Correlation
plt.matshow(df1.corr())

plt.title("Baseball Corelation HeatMap")


# In[22]:


#Plotting all values linearly
df1.plot()
plt.xlabel('Year')
plt.title("MLB Revenue & Stats")


# In[23]:


# Plotting HR vs. Revenue
df1['Home_Run_Avg'].plot(color='Red', linestyle='solid', linewidth = 2, label = "Homerun Avg")
df1['Revenue_in_B'].plot(color='Blue', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='Yellow', markersize=8, label = "Revenue")

plt.xlabel('Year')
plt.legend()
plt.title("HomeRuns & Revenue in Billions")


# In[24]:


# Plotting AVG K vs. Revenue
df1['Avg_K'].plot(color='Orange', linestyle='solid', linewidth = 2, label = "Avg K %")
df1['Revenue_in_B'].plot(color='Blue', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='Yellow', markersize=8, label = "Revenue")

plt.xlabel('Year')
plt.legend()
plt.title("Avg K % & Revenue in Billions")


# In[25]:


# Plotting AVG Walk vs. Revenue
df1['Avg_Walk'].plot(color='lightskyblue', linestyle='solid', linewidth = 2, label = "Avg Walk %")
df1['Revenue_in_B'].plot(color='Blue', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='Yellow', markersize=8, label = "Revenue")

plt.xlabel('Year')
plt.legend()
plt.title("Avg Walk % & Revenue in Billions")


# In[26]:


# Plotting AVG BA vs. Revenue
df1['Avg_BA'].plot(color='springgreen', linestyle='solid', linewidth = 2, label = "Avg BA")
df1['Revenue_in_B'].plot(color='Blue', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='Yellow', markersize=8, label = "Revenue")

plt.xlabel('Year')
plt.legend()
plt.title("Player Batting Avg & Revenue in Billions")


# In[27]:


# Plotting AVG SM vs. Revenue
df1['Avg_SM'].plot(color='fuchsia', linestyle='solid', linewidth = 2, label = "Avg Swing & Miss")
df1['Revenue_in_B'].plot(color='Blue', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='Yellow', markersize=8, label = "Revenue")

plt.xlabel('Year')
plt.legend()
plt.title("Avg Swing & Miss & Revenue in Billions")


# In[28]:


# Plotting AVG Exit Velo vs. Revenue
df1['Avg_ExitVelo'].plot(color='chocolate', linestyle='solid', linewidth = 2, label = "Avg Exit Velo")
df1['Revenue_in_B'].plot(color='Blue', linestyle='dashed', linewidth = 2,
         marker='o', markerfacecolor='Yellow', markersize=8, label = "Revenue")

plt.xlabel('Year')
plt.legend()
plt.title("Avg Exit Velo & Revenue in Billions")


# # Linear Regression ML

# In[30]:


df1


# In[31]:


#Check for Duplicates in df1
df1.duplicated().any()


# In[34]:


#Check for Outliers in df1
fig, axs = plt.subplots(4, figsize = (5,5))
plt1 = sns.boxplot(df1['Home_Run_Avg'], ax = axs[0])
plt2 = sns.boxplot(df1['Avg_K'], ax = axs[1])
plt3 = sns.boxplot(df1['Avg_Walk'], ax = axs[2])
plt4 = sns.boxplot(df1['Avg_SM'], ax = axs[3])
plt.tight_layout()


# In[36]:


#Looking at Distribution of Revenue 
sns.distplot(df1['Revenue_in_B']);


# In[40]:


# Relation of Revenue to x-variables 
sns.pairplot(df1, x_vars=['Home_Run_Avg', 'Avg_K', 'Avg_Walk', 'Avg_SM'], y_vars='Revenue_in_B', height=4, aspect=1, kind='scatter')
plt.show()


# In[42]:


#Correlation Heat Map
sns.heatmap(df1.corr(), annot = True)
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[44]:


#Setting the value for X and Y
x = df1[['Home_Run_Avg']]
y = df1['Revenue_in_B']


# In[45]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)


# In[46]:


slr= LinearRegression()  
slr.fit(x_train, y_train)


# In[47]:


#Printing the model coefficients
print('Intercept: ', slr.intercept_)
print('Coefficient:', slr.coef_)


# In[1]:


#Line of best fit
#plt.scatter(x_train, y_train)
#plt.plot(x_train, 4.482 + 0.231*x_train, 'r')
#plt.show()


# In[51]:


#Prediction of Test and Training set result  
y_pred_slr= slr.predict(x_test)  
x_pred_slr= slr.predict(x_train) 


# In[52]:


print("Prediction for test set: {}".format(y_pred_slr))


# In[53]:


#Actual value and the predicted value
slr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_slr})
slr_diff


# In[54]:


#Predict for any value
slr.predict([[56]])


# In[55]:


# print the R-squared value for the model
from sklearn.metrics import accuracy_score
print('R squared value of the model: {:.2f}'.format(slr.score(x,y)*100))


# Findings: 58.67% of the data fit the regression model

# In[56]:


# Model fit for 0
meanAbErr = metrics.mean_absolute_error(y_test, y_pred_slr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_slr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_slr))

print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


# Conclusion:
# 
# There is not a large enough sample size to accurately predict Revenue
