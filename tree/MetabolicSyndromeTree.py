# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 19:59:37 2022

@author: niko-_000
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:/Users/SeuUsuario/PastaDeDestino/MetabolicSyndrome.csv'

data = pd.read_csv(path)
data.describe()
#features = data.drop('MetabolicSyndrome', axis = 1)
features = data.drop(['Marital','MetabolicSyndrome','Income','WaistCirc','BMI'], axis = 1)
features.shape
features.columns
print (features)
labels = np.array(data['MetabolicSyndrome'])
print(labels.shape)
print(labels)

#tratando dados faltantes
for name_col in features.columns:
    print("Number of Nan on column", name_col, ";", features[name_col].isnull().sum())

features = pd.get_dummies(features)

features.head()    
features.columns


  
#treino e teste

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25, random_state=42)
train_labels.shape
train_features.shape
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(train_features, train_labels);
print(features)

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

predict = dt.predict(test_features)
predict.shape

print("R2:", dt.score(test_features, test_labels))

from sklearn import tree
import graphviz

features_list = list(features.columns)
print(features_list)
print(len(features_list))

dot_data = tree.export_graphviz(dt, out_file=None,feature_names=features_list,filled=True, rounded=True)

graph = graphviz.Source(dot_data, format='png')

dt_path = 'C:/Users/SeuUsuario/PastaDeDestino/'
graph.render(dt_path)
