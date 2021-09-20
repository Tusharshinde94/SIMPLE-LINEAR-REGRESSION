#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[4]:


Sal_h=pd.read_csv("E:\TUSHAR\ASSINMENT DONE BY TUSHAR\ASSIGNMENT NO 4 (SIMPLE LINEAR REGRESSION)\Salary_Data.csv")


# In[5]:


Sal_h.head()


# In[11]:


Sal_h.describe()


# In[16]:


Sal_h.plot(x='YearsExperience', y='Salary', style='o') 
plt.title('Salaryhike vs YearofExperience')  
plt.xlabel('Salary') 
plt.ylabel('YearsofExp')
plt.show()


# In[25]:


# calculate Pearson's correlation
from scipy.stats import pearsonr
corr, _ = pearsonr(Sal_h['Salary'], Sal_h['YearsExperience'])
print('Pearsons correlation: %.3f' % corr)


from scipy.stats import spearmanr
# calculate spearman's correlation
corr, _ = spearmanr(Sal_h['Salary'], Sal_h['YearsExperience'])
print('Spearmans correlation: %.3f' % corr)


# In[26]:


import seaborn as sns
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(Sal_h['Salary'])
plt.show() 


# In[30]:


plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(Sal_h['YearsExperience'])
plt.show()


# In[32]:


# Input dataset
X = Sal_h['Salary'].values.reshape(-1,1)
print(X)
# Output or Predicted Value of data
y = Sal_h['YearsExperience'].values.reshape(-1,1)
#print(log(y))


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state =42)


# In[35]:


predict_reg = LinearRegression()


# In[36]:


predict_reg.fit(X_train, y_train)


# In[37]:


print(" Intercept value of Model is " ,predict_reg.intercept_)
print("Coefficient value of Model is ", predict_reg.coef_)


# In[38]:


y_pred = predict_reg.predict(X_test)


# In[39]:


pmsh_pf = pd.DataFrame({'Actual':y_test.flatten(), 'Predict': y_pred.flatten()})
pmsh_pf


# In[40]:


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


# In[41]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R^2 Score :          ", metrics.r2_score(y_test, y_pred))


# In[ ]:


#END WITH ASSIGNMENT
#TUSHAR SHINDE

