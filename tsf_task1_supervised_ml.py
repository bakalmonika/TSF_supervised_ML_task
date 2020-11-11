#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 
# # AUTHOR: MONIKA BAKAL
# 
# ### Task 1: Using Supervised ML find the score based on number of hours of study i.e for 9.25 hours of study

# In[152]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[153]:


# importing data set
# if you have the data set downloaded then use: 
# dataset=pd.read_csv(r'location of file\filename.csv')
dataset = pd.read_csv("http://bit.ly/w-data")


# In[154]:


dataset.head()


# In[155]:


dataset.describe()


# In[156]:


X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values  


# ### vislualizing the given data
# 

# In[157]:


# plotting given data
plt.scatter(X,Y)
plt.title("original data")
plt.xlabel("Hours")
plt.ylabel("score")
plt.show()


# ### splitting the dataset into training set and test set
# 

# In[158]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### visualising traininng dataset
# 

# In[159]:


plt.scatter(X_train, y_train)
plt.title("training dataset")


# ### visualising test set
# 

# In[160]:


plt.scatter(X_test,y_test)
plt.title('test set data')


# ### Training Simple linear regression model 

# In[161]:


from sklearn.linear_model import LinearRegression  
lm = LinearRegression()  
lm.fit(X_train, y_train) 

print("Training.")


# In[162]:


predict = lm.predict(X_test)


# In[163]:


print(predict)


# In[164]:


print(lm.intercept_)


# In[165]:


print(lm.coef_)


# In[166]:


# plotting the graph with fitting
plt.scatter(X,y, color="green")
plt.plot(X_test,predict,color="blue")


# ### Predicting the value or score
# 

# In[167]:


print('score for 9.25 hrs study is', lm.predict([[9.25]]))


# In[168]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, predict)) 


# In[ ]:




