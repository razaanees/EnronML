#!/usr/bin/python

import sys, pandas as pd, numpy as np
import pickle
from pprint import pprint
sys.path.append("../tools/")
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

def select_features(feature_list, my_dataset, k):
    '''
    Select k number of features based on SelectKBest function and
    StratifiedShuffleSplit

    feature_list = list of strings representing feature names
    my_dataset = dataset containing all features and labels
    k = number of desired features
    '''
    from sklearn.model_selection import StratifiedShuffleSplit

    # Create feature and label arrays from the dataset
    data = featureFormat(my_dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    # create sss with 1000 splits
    sss = StratifiedShuffleSplit(n_splits=1000, random_state=42)
    feature_scores = {}

    # Create 1000 different sets of training and testing samples
    for train, test in sss.split(features, labels):
        features_train = [features[i] for i in train]
        labels_train = [labels[i] for i in train]

        # fit the selectkbest function on each set of training data
        selector = SelectKBest(k=k)
        selector.fit(features_train, labels_train)

        # Get list of features, scores, and pvalues for each selector
        feature_indices = selector.get_support(indices=True)
        sel_features = [(feature_list[i+1], selector.scores_[i], selector.pvalues_[i])
                        for i in feature_indices]

        # Gather the scores and pvalue of each feature from each split
        for feat, score, pval in sel_features:
            if feat not in feature_scores:
                feature_scores[feat] = {"scores": [], "pvalue": []}
            feature_scores[feat]['scores'].append(score)
            feature_scores[feat]['pvalue'].append(pval)

    # Get average score and pvalue of each feature
    feature_scores_l = []
    for feat in feature_scores:
        feature_scores_l.append((
        feat,
        np.mean(feature_scores[feat]['scores']),
        np.mean(feature_scores[feat]['pvalue'])
        ))

    import operator
    sorted_feature_scores = sorted(feature_scores_l, key=operator.itemgetter(1),
                                    reverse = True)
    sorted_feature_scores_str = ["{}: {} {}".format(z[0], z[1], z[2])
                                 for z in sorted_feature_scores]

    print "feature: score, p-value"
    for line in sorted_feature_scores_str:
        print line
    return


features_list = ['poi', 'salary', 'bonus',
                 'exercised_stock_options', 'total_stock_value',
                 'deferred_income']

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

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
my_dataset = data_dict2

# Get list of all features so the best ones can be chosen using the
# select_feature function
# Uncomment the code below to see feature f-scores determined by SelectKBest
'''
leave_out = ['email_address', 'poi']
all_feats = df.columns.values
new_all_feats = np.array([x for x in all_feats if x not in leave_out])
all_feats = np.insert(new_all_feats, 0, 'poi')


# display the features with the highest scores
select_features(all_feats, my_dataset, 6)
'''
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
          }
scaler = StandardScaler()
pipe = Pipeline(steps=[('scaler', scaler), ('svm', svm)])
'''
# Tried decision tree classifier with many parameters
'''
tree = DecisionTreeClassifier()
params = {'tree__criterion': ('gini', 'entropy'),
              'tree__splitter': ('best', 'random'),
              'tree__min_samples_split': [2, 10, 20],
              'tree__max_depth': [10, 15, 20, 25, 30],
              'tree__max_leaf_nodes': [5, 10, 30]}
pipe = Pipeline(steps=[('tree', tree)])
'''
'''
cv = StratifiedShuffleSplit(labels, 100, random_state=42)
gs = GridSearchCV(pipe, param_grid=params, cv=cv)
gs.fit(features, labels)
clf = gs.best_estimator_
'''
# Simple Naive Bayes gives the best metric scores
clf = GaussianNB()

dump_classifier_and_data(clf, my_dataset, features_list)
