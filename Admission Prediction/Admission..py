import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import math

ss =['LinearRegression', 'LogisticRegression', ]
class CsvOperation:
    global data, LM

    def __init__(self, name):
        self.name = name

    def dataSet(self):
        data = pd.read_csv(self.name)
        data.drop(['Serial No.'], inplace=True, axis=1)

    def training(self):
        length = math.ceil(len(data) * 0.22)
        xTrain = data.drop(['Chance of Admit'], axis=1)[:length]
        yTrain = data.drop(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'],
                           axis=1)[:length]
        LM = LinearRegression()
        LM.fit(xTrain, yTrain)

    def Testing(self):
        xTest = data.drop(['Chance of Admit'], axis=1)[301:]
        yTest = data.drop(['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research'], axis=1)[
                301:]
        yPredicted = LM.predict(xTest)
        e = mean_absolute_error(yTest, yPredicted)
        print(e)
