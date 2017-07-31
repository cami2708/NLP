from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

data=pd.read_csv('spambase.txt').as_matrix()
np.random.shuffle(data)

x=data[:, :48]
y=data[:, -1]

xTrain=x[:-100,]
yTrain=y[:-100,]
xTest=x[-100:,]
yTest=y[-100:,]

model=MultinomialNB()
model.fit(xTrain, yTrain)
print "Classification rate for Naive Bayes:", model.score(xTest, yTest)
