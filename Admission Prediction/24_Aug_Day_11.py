
import pandas as pd
df=pd.read_csv("Admission_Predict.csv")


# In[3]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


print(df['GRE Score'].min())


# In[4]:


#Data Preprocessing
#Chance of admit is not dependent on serial number, therefore we do not need serial number
#for ML purpose

df.drop(['Serial No.'],inplace=True,axis=1)


# In[5]:


df.head()


# In[ ]:


print(df['Chance of Admit'])


# In[ ]:


print(df.columns)


# In[ ]:


print(df['Chance of Admit'])


# In[6]:


#df.rename(columns={'Chance of Admit ': 'Chance of Admit'},inplace=True)


# In[ ]:


print(df.columns)


# In[7]:


#X is feature on which outcome y is dependent
print(df.shape)
X=df.drop(['Chance of Admit'],axis=1)
print(X.shape)


# In[ ]:


X.head()


# In[8]:


#y is outcome which we want to predict through Machine Learning
y=df['Chance of Admit']
print(y)


# In[9]:


#Now X (Features) and y (Outcome) is ready
#Now split the data into two parts: train data(320 rows) and test data(80 rows)
#Generally testing data is kept as 20%(0.20) to 30%(0.30)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.20)


# In[10]:


X_train.shape


# In[23]:


X_train.columns


# In[11]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[12]:


#Only these three lines in python will create a Machine Learning
#Model for you
from sklearn.linear_model import LinearRegression
m=LinearRegression()
m.fit(X_train,y_train)

#m object is a ML Model, which can predict outcome, if feature is given
# as input to this


# In[13]:


y_predict=m.predict(X_test)


# In[ ]:


#y_test is actual outcome directly taken from sample data
#y_predict is predicted outcome by the ML model i.e. m in this case
df1=pd.DataFrame({"Actual": y_test,"Predicted":y_predict })
df1.to_csv("04_Sep_Output.csv")


# In[14]:


#y_test is actual outcome directly taken from sample data
#y_predict is predicted outcome by the ML model i.e. m in this case
df1=pd.DataFrame({"Actual": y_test,"Predicted":y_predict })
df1['error']=df1['Actual']-df1['Predicted']
df1['error']=df1['error'].abs()
df1.to_csv("04_Sep_Output.csv")


# In[15]:


df1.head()


# In[16]:


Mean_Absolute_Error=df1['error'].mean()
print(Mean_Absolute_Error)


# In[ ]:


#4.8% is the error of your ML model created using Linear Regression
#Algorithm
#why Error is modified when data is same?


# In[17]:


from sklearn.metrics import mean_absolute_error
e=mean_absolute_error(y_test,y_predict)
print(e)


# In[27]:


#data of a new user or current user who wants to predict the outcome(chance of his admission)
#Deployment of a Machine Learning model
gre=int(input("What is your GRE Score (between 290 to 340):"))
toefl=int(input("What is your TOEFL Score (between 90 to 120):"))
univ=int(input("What is your University Rating ( 1 to 5 ):"))
sop=float(input("Rate your Statement of Purpose ( 1 to 5):"))
lor=float(input("What is strength of  your Letter of Recommendation ( 1 to 5) :"))
cgpa=float(input("What is your CGPA ( 6 to 10):"))
research=int(input("Do You have Research Experience (Enter 0 for No and 1 for Yes:"))

#Very important: the sequence of new data will be exactly same as per training data columns i.e.
#X_train columns
list=[gre,toefl,univ,sop,lor,cgpa,research]

#predict function takes input argument i.e. feature data as Data Frame which may consist
#of many rows or records or entity data

Newdf=pd.DataFrame([list])
y_p=m.predict(Newdf)

print("Your Chance of admission is:",y_p[0]*100-4.81,"  to  ",y_p[0]*100+4.81," Percent")


# In[24]:


y_predict=m.predict(X_train)


# In[ ]:


print(y_predict)


# In[26]:


from sklearn.metrics import mean_absolute_error
e=mean_absolute_error(y_train,y_predict)
print(e)


# In[ ]:


#Finished entire process of machine learning for this particular problem in
#a simple way

#How can you say that this algorithm is better
#only by experiments
#you make a model using all possible algorithms and test that, find error
#whatever has minimum error, that algo is suitable for your problem.
#Apply algorithm depending on type of problem eg. regression in this case


# In[28]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
#rfr is the model on basis of this algorithm


# In[31]:


yr_predict=rfr.predict(X_test)


# In[32]:


from sklearn.metrics import mean_absolute_error
e=mean_absolute_error(y_test,yr_predict)
print(e)


# In[33]:


#Algorithm Name: Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)
#dtr is a ML Model


# In[34]:


ydtr_predict=dtr.predict(X_test)


# In[35]:


from sklearn.metrics import mean_absolute_error
e=mean_absolute_error(y_test,ydtr_predict)
print(e)


# In[ ]:


# We have applied three regression algorithms
#We found that error is lowest in case of Linear Regression Algorithm
#that means m is best model for my problem
#therefore final decision is that I will depoly m as my ML Model
#for this model


# In[ ]:


#how many algorithms are there
#No one knows why? bcoz research is going on......
#what you know is that it is a regression problem
#So I have to apply regression algorithm


# In[37]:


#Algoirthm Name: Support Vector Regression
from sklearn.svm import SVR
s = SVR()
s.fit(X_train,y_train)


# In[38]:


ys_predict=s.predict(X_test)
from sklearn.metrics import mean_absolute_error
e=mean_absolute_error(y_test,ys_predict)
print(e)


# In[43]:


#Algorithm: Bayesian Ridge Regression is little better than linear regression
from sklearn.linear_model import BayesianRidge
br = BayesianRidge()
br.fit(X_train,y_train)


# In[44]:


ys_predict=br.predict(X_test)
from sklearn.metrics import mean_absolute_error
e=mean_absolute_error(y_test,ys_predict)
print(e)


# In[ ]:




