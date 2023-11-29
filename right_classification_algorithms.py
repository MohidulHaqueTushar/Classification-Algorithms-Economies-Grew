# -*- coding: utf-8 -*-
""" Script to run three different classification algorithms for explaining
    whether region's economies grew by more than 5% based on the data provided. 
    Standard goodness measures for classification algorithms also included.
    
Author: Md Mohidul Haque
"""

""" Import modules """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
import time

""" Load and prepare the data """
df_explaining_vars = pd.read_csv("OECD_data_explaining_vars.csv", header=0, index_col=0)
df_data_growth = pd.read_csv("OECD_data_growth.csv", header=0, index_col=0)

"""Information about data: df_explaining_vars"""
print("The dimensions of the data 'df_explaining_vars': ",df_explaining_vars.shape)

"""Information about data: df_data_growth"""
print("The dimensions of the data 'df_data_growth':", df_data_growth.shape)

"""Check if the data is imbalanced"""
print("\nChecking if the data is imbalanced:")
count_of_classes = df_data_growth["Growth > 5%"].value_counts()
class_percentages = count_of_classes / count_of_classes.sum() # Calculate class percentages
print("\nClass Counts:")
print(count_of_classes)
print("\nClass Percentages:")
print(class_percentages)

""" Split test and training set: Stratified sampling"""
x_train, x_test, y_train, y_test = train_test_split(df_explaining_vars, df_data_growth, test_size=0.2, stratify=df_data_growth, random_state=42)

"""Key for three classification algorithms"""
list_of_classifiers = {
    'DecisionTree': DecisionTreeClassifier(max_depth=15),
    'RandomForest': RandomForestClassifier(),
    'SupportVectorMachine': SVC(kernel='linear', C=0.0025),
    }

print("\nUsing three classification algorithms:")
for key in list_of_classifiers.keys():
    print("\nCommencing classification with: ", key)
    """ Timing of classifiers """
    starttime = time.time()
    
    """ Define a classifier"""
    classifier = list_of_classifiers[key]
    
    """ Fit classifier"""
    fit = classifier.fit(x_train, np.asarray(y_train).ravel())
    
    """ Predict test set output """
    prediction = classifier.predict(x_test)
    
    time_elapsed = time.time() - starttime 
    print("Time elapsed: ", time_elapsed, "seconds")
    
    """ Assess performance"""
    print("Precision: ", sklearn.metrics.precision_score(y_test, prediction))
    print("Recall: ", sklearn.metrics.recall_score(y_test, prediction))
    print("F1: ", sklearn.metrics.f1_score(y_test, prediction))
    print("Accuracy: ", sklearn.metrics.accuracy_score(y_test, prediction))