#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("D:\\BBAU1\\Admission_Predict.csv")


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


print(df['GRE Score'].min())


# In[7]:


#Data Preprocessing
#Chance of admit is not dependent on serial number, therefore we do not need serial number
#for ML purpose

df.drop(['Serial No.'],inplace=True,axis=1)


# In[8]:


df.head()


# In[9]:


print(df['Chance of Admit'])


# In[10]:


print(df.columns)


# In[11]:


print(df['Chance of Admit '])


# In[12]:


df.rename(columns={'Chance of Admit ': 'Chance of Admit'},inplace=True)


# In[13]:


print(df.columns)


# In[14]:


#X is feature on which outcome y is dependent
print(df.shape)
X=df.drop(['Chance of Admit'],axis=1)
print(X.shape)


# In[16]:


X.head()


# In[17]:


#y is outcome which we want to predict through Machine Learning
y=df['Chance of Admit']
print(y)


# In[29]:


#Now X (Features) and y (Outcome) is ready
#Now split the data into two parts: train data(320 rows) and test data(80 rows)
#Generally testing data is kept as 20%(0.20) to 30%(0.30)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20)


# In[30]:


X_train.shape


# In[31]:


X_test.shape


# In[32]:


y_train.shape


# In[33]:


y_test.shape


# In[34]:


#Only these three lines in python will create a Machine Learning
#Model for you
from sklearn.linear_model import LinearRegression
m=LinearRegression()
m.fit(X_train,y_train)

#m object is a ML Model, which can predict outcome, if feature is given
# as input to this


# In[35]:


y_predict=m.predict(X_test)


# In[36]:


#y_test is actual outcome directly taken from sample data
#y_predict is predicted outcome by the ML model i.e. m in this case
df1=pd.DataFrame({"Actual": y_test,"Predicted":y_predict })
df1.to_csv("04_Sep_Output.csv")


# In[38]:


#y_test is actual outcome directly taken from sample data
#y_predict is predicted outcome by the ML model i.e. m in this case
df1=pd.DataFrame({"Actual": y_test,"Predicted":y_predict })
df1['error']=df1['Actual']-df1['Predicted']
df1['error']=df1['error'].abs()
df1.to_csv("04_Sep_Output.csv")


# In[39]:


df1.head()


# In[40]:


Absolute_Mean_Error=df1['error'].mean()
print(Absolute_Mean_Error)


# In[41]:


#4.6% is the error of your ML model created using Linear Regression
#Algorithm


# In[42]:


from sklearn.metrics import mean_absolute_error
e=mean_absolute_error(y_test,y_predict)
print(e)


# In[ ]:




