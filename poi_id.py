#!/usr/bin/python

import sys, pandas as pd, numpy as np
import pickle
from pprint import pprint
sys.path.append("../tools/")
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'deferred_income',
                 'exercised_stock_options', 'total_stock_value']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
# Firstly, let's identify outliers. This was done by converting the dictionary
# into a pandas dataframe for easy EDA.
all_values = list(data_dict.values())
df = pd.DataFrame.from_records(all_values)
names = pd.Series(list(data_dict.keys()))
df.set_index(names, inplace=True)
df.replace(to_replace='NaN', value=np.nan, inplace=True)
df = df.apply(lambda x: pd.to_numeric(x, errors='ignore')).fillna(0)
df = df.drop(['TOTAL', 'WHALEY DAVID A', 'WROBEL BRUCE', 'LOCKHART EUGENE E',
              'THE TRAVEL AGENCY IN THE PARK', 'GRAMM WENDY L'])
df['convo_with_poi'] = (df['from_this_person_to_poi'])/(df['to_messages']).fillna(0)
df = df.fillna(0)
data_dict2 = df.to_dict(orient='index')

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
my_dataset = data_dict2

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Tried SVC with different parameters
'''
svm = SVC()
params = {'svm__C': [1.0, 100.0],
          'svm__kernel': ('poly', 'rbf', 'sigmoid'),
          'svm__gamma': [0.1, 0.0001, 0.01, 0.001],
          'svm__class_weight': [{1:2}, {1:10}, {1:5}]
          }'''
# Tried decision tree classifier with many parameters
'''
tree = DecisionTreeClassifier()
parameters = {'tree__criterion': ('gini', 'entropy'),
              'tree__splitter': ('best', 'random'),
              'tree__min_samples_split': [2, 10, 20],
              'tree__max_depth': [10, 15, 20, 25, 30],
              'tree__max_leaf_nodes': [5, 10, 30]}
scaler = MinMaxScaler()
cv = StratifiedShuffleSplit(labels, 100, random_state=42)
pipe = Pipeline(steps=[('scaler', scaler), ('tree', tree)])
gs = GridSearchCV(pipe, param_grid=parameters, cv=cv)
gs.fit(features, labels)
clf = gs.best_estimator_
'''
# Simple Naive Bayes gives the best metric scores
clf = GaussianNB()


dump_classifier_and_data(clf, my_dataset, features_list)
