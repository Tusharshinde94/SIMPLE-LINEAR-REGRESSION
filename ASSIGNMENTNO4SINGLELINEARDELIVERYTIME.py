#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


delv_data=pd.read_csv("E:\TUSHAR\ASSINMENT DONE BY TUSHAR\ASSIGNMENT NO 4 (SIMPLE LINEAR REGRESSION)\delivery_time.csv")


# In[3]:


delv_data.head()


# In[4]:


delv_data=delv_data.rename(columns={'Delivery Time': 'dt','Sorting Time': 'st' })


# In[5]:


delv_data.head()


# In[6]:


delv_data.corr()


# In[7]:


plt.scatter(x=delv_data.st, y=delv_data.dt, color='RED')
plt.xlabel("Sorting time")
plt.ylabel("Delivery time")


# In[20]:


plt.boxplot(delv_data.dt)


# In[8]:


plt.hist(delv_data.dt, bins=6)


# In[11]:


model2=smf.ols("dt~st",data=delv_data).fit()


# In[12]:


model2.params


# In[13]:


model2.summary()


# In[14]:


model2.conf_int(0.05)


# In[20]:


model3=smf.ols("dt~np.log(st)",data=delv_data).fit()


# In[21]:


model3.params


# In[23]:


model3.summary()


# In[22]:


model2.conf_int(0.05) # 95% confidence interval


# In[24]:


model3.conf_int(0.05) # 95% confidence interval


# In[26]:


pred2 = model2.predict(delv_data) # Predicted values of dt using the model


# In[27]:


pred3 = model3.predict(delv_data) # Predicted values of dt using the model


# In[28]:


plt.scatter(x=delv_data.st, y=delv_data.dt, color='green')
plt.plot(delv_data.st, pred2,color='black')
plt.xlabel("Sorting time")
plt.ylabel("Delivery time")


# In[29]:


plt.scatter(x=delv_data.st, y=delv_data.dt, color='red')
plt.plot(delv_data.st, pred3,color='green')
plt.xlabel("Sorting time")
plt.ylabel("Delivery time")


# In[ ]:


#Thanks Assignment Completed Delivery_time
#Question :- Predict delivery time using sorting time
#Tushar Shinde 23rd Aug 2021

