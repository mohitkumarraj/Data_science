# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:59:19 2021

@author: mk626
"""

import pandas as pd
import numpy as np
import os
from sklearn import preprocessing 
import pickle
    # from sklearn.preprocessing import LabelEncoder

os.chdir('PycharmProjects\income_classifier') 
df = pd.read_csv('income.csv',na_values=['?','??'])     
df.head()

print(np.unique(df['hours-per-week']))

df=df.drop(['fnlwgt','educational-num'],axis=1)
df.columns
df.dropna(axis=0,inplace=True)


data=df.copy(deep=True)

data['marital-status'].value_counts()
data.replace(['Divorced', 'Married-AF-spouse',  
              'Married-civ-spouse', 'Married-spouse-absent',  
              'Never-married', 'Separated', 'Widowed'],['divorced', 'married', 'married', 'married', 
              'not married', 'not married', 'not married'],inplace=True)

data.describe(include='object')  
                                                      
category_col=data.select_dtypes(exclude='int64')
category_col_name=list(category_col.columns)
category_col_name

labelEncoder = preprocessing.LabelEncoder() 
  


mapping_dict ={} 
for col in category_col_name: 
    df[col] = labelEncoder.fit_transform(df[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
    mapping_dict[col]= le_name_mapping 

print(mapping_dict) 

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 

X = df.values[:, 0:12] 
Y = df.values[:, 12] 

X_train, X_test, y_train, y_test = train_test_split( 
           X, Y, test_size = 0.3, random_state = 100) 

dt_clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                     random_state = 100, 
                                     max_depth = 5, 
                                     min_samples_leaf = 5) 



dt_clf_gini.fit(X_train, y_train) 
y_pred_gini = dt_clf_gini.predict(X_test) 

pickle.dump(dt_clf_gini, open('model.pkl','wb'))

print ("Desicion Tree using Gini Index\nAccuracy is ", 
             accuracy_score(y_test, y_pred_gini)*100 ) 