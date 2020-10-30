import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


data = pd.read_csv('Admission_Predict.csv')
data.drop(['Serial No.'], inplace=True, axis=1)

xTrain = data.drop(['Chance of Admit'], axis=1)[:380]
yTrain = data.drop(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'], axis=1)[:380]

xTest = data.drop(['Chance of Admit'], axis=1)[301:]
yTest = data.drop(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'], axis=1)[301:]

LM = LinearRegression()
LM.fit(xTrain, yTrain)

yPredicted = LM.predict(xTest)

#my = pd.DataFrame({'actual': yTest, 'Predicted': yPredicted})

e=mean_absolute_error(yTest, yPredicted)
print(e)

