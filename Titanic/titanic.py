#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 22:19:54 2017
@author: yogi
"""
#Classification
#Import the packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb
color = sns.color_palette()
#%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

#importing dataset
dataset = pd.read_csv('train.csv')

#Know your data
print("Train variants shape : ",dataset.shape)
print("Header : ",dataset.columns)

print("---------------------------------")
print("Records")
print(dataset.head())

print("-----------------------------------------------")
print("Description")
print(dataset.describe())

print("--------------------------------------------------")
print("Info")
print(dataset.info())

#Create corelation matrix
# Compute the correlation matrix
corr = dataset.corr()
plt.figure(figsize=(8,8))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(corr, linewidths=0.1, vmax=1.0, 
            square=True, linecolor='white', annot=True)

#Graphical representation
#histogram
sns.distplot(dataset['Fare']);
#skewness and kurtosis
print("Skewness: ",dataset['Fare'].skew())
print("Kurtosis: ",dataset['Fare'].kurt())

#scatter plot Fare/Survived
var = 'Fare'
data = pd.concat([dataset['Survived'], dataset[var]], axis=1)
data.plot.scatter(x=var, y='Survived', ylim=(0,2));

#box plot Fare/Survived
var = 'Fare'
data = pd.concat([dataset['Survived'], dataset[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="Survived", data=data)
fig.axis(ymin=0, ymax=2);

#Pair Plot
sns.pairplot(dataset, hue='Age')

#missing data
total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#Data Processing
#Remove the useless data
data_df = dataset.drop(['PassengerId','Survived','Name','Ticket'], axis = 1)

#transform data
#Change pclass to Upper/Middle/Lower

#Change sibsp to 0/1
for index, row in data_df.iterrows():
    if data_df.loc[index, "SibSp"] != 0:
        #print(data_df.loc[index, "SibSp"])
        data_df.loc[index,'SibSp'] = 1


#Change parch to 0/1
for index, row in data_df.iterrows():
    if data_df.loc[index, "Parch"] != 0:
        #print(data_df.loc[index, "Parch"])
        data_df.loc[index,'Parch'] = 1
        
#Change cabin to 0/1
for index, row in data_df.iterrows():
    #print(isinstance(data_df.loc[index,'Cabin'], str))
    if isinstance(data_df.loc[index,'Cabin'], str):
        #print(data_df.loc[index, "SibSp"])
        data_df.loc[index,'Cabin'] = float(1)
    else:
        data_df.loc[index,'Cabin'] = float(0)

print(data_df.describe())

#Finding data which can be categorical 
for f in data_df.columns:
    if data_df[f].dtype=='object' and dataset[f].name != 'Cabin':
        #print(dataset[f].name)
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data_df[f].values)) 
        data_df[f] = lbl.transform(list(data_df[f].values))

print(data_df.head())
print(data_df.describe())
#A = dataset.loc[:,['Cabin','Fare']] 

#Creating independent matrix
#X = dataset.drop(['Survived','Name','Ticket','Cabin','PassengerId'], axis = 1)
#print(X.head())
X = data_df.values

#Creating dependent vector
Y = dataset.loc[:,'Survived'].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 2:3])
X[:, 2:3] = imputer.transform(X[:, 2:3])

#Splitting the data set into training and test set
from sklearn.cross_validation import train_test_split
x_Train, x_Test, y_Train, y_Test = train_test_split(X, Y, test_size = 0.25, 
                                                    random_state = 0)

#Feature Scaling
#A must do when algo is based on eucledian distance
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_Train = sc_X.fit_transform(x_Train)
x_Test = sc_X.transform(x_Test)

#Fitting Classifier to Training set
from sklearn.ensemble import RandomForestClassifier 
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', 
                                    random_state = 0)
classifier.fit(x_Train,y_Train)

#Predicting the Test set results. Vector of prediction
y_Pred = classifier.predict(x_Test)

#Evaluating the predictions of model
#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_Pred)
print(cm)

xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'min_child_weight':1,
    'silent': 1,
    'seed':0
}

xgtrain = xgb.DMatrix(x_Train,y_Train, feature_names=data_df.columns)
xgtest = xgb.DMatrix(x_Test,y_Test, feature_names=data_df.columns)
watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
num_rounds = 200 # Increase the number of rounds while running in local
model = xgb.train(xgb_params, xgtrain, num_rounds, 
                  watchlist, early_stopping_rounds=50, verbose_eval=5)

# plot the important features #
fig, ax = plt.subplots(figsize=(8,8))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show()

