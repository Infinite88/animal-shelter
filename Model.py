
# coding: utf-8

# In[54]:

from __future__ import division
from __future__ import absolute_import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.learning_curve import learning_curve
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score,log_loss, f1_score
from sklearn.grid_search import GridSearchCV 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings(u'ignore')


# In[18]:

# Importing cleaned dataset for both the testing set and training set 
trainDF = pd.read_csv(u'files/cleanTrain.csv')
testDF = pd.read_csv(u'files/cleanTest.csv')


# In[19]:

trainDF.head()


# In[20]:

trainDF.info()


# In[21]:

testDF.info()


# In[22]:

# Dropping features so that datasets features match
train = trainDF.drop([u'AnimalID', u'OutcomeType'], axis = 1)
test = testDF.drop(u'ID', axis=1)


# In[23]:

train.info()


# In[24]:

test.info()


# In[25]:

# Transform data into numeric values
lb = LabelEncoder()
categorical_columns = train.columns[train.dtypes == u'object']
for var in categorical_columns:
    full_data = pd.concat((train[var],test[var]),axis=0).astype(u'str')
    lb.fit(full_data )
    train.loc[:, var] = lb.transform(train[var].astype(u'str'))
    test.loc[:, var] = lb.transform(test[var].astype(u'str'))


# In[26]:

train.head()


# In[27]:

#
X_all = train.values
y_all = trainDF.OutcomeType.values

x_test = test.values


# In[28]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all,
                                                                     test_size=0.30, random_state=67) 


# In[29]:

def train_classifier(clf, X_train, y_train):
    print u"Training {}...".format(clf.__class__.__name__)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print u"Done!\nTraining time (secs): {:.3f}".format(end - start)
    
def predict_labels(clf, X_train, y_train):
    print u"Predicting labels using {}...".format(clf.__class__.__name__)
    start = time.time()
    y_pred = clf.predict(X_train)
    end = time.time()
    print u"Done!\nPrediction time (secs): {:.3f}".format(end - start)
    return cross_val_score(clf, X_train, y_train, scoring= u'log_loss')

def train_predict(clf, X_train, y_train, X_test, y_test):
    print u"------------------------------------------"
    print u"Training set size: {}".format(len(X_train))
    train_classifier(clf, X_train, y_train)
    print u"Log_loss score for training set: {}".format(predict_labels(clf, X_train, y_train))
    print u"Log_loss for test set: {}".format(predict_labels(clf, X_test, y_test))


# In[30]:

# Loading the classifiers that will be used
clf = GaussianNB()
rfc = RandomForestClassifier()
gbc = GradientBoostingClassifier()
knn = KNeighborsClassifier()
log = LogisticRegression()


# In[31]:

# The training of each model
train_predict(clf, X_train, y_train, X_test, y_test)
train_predict(rfc, X_train, y_train, X_test, y_test)
train_predict(gbc, X_train, y_train, X_test, y_test)
train_predict(knn, X_train, y_train, X_test, y_test)
train_predict(log, X_train, y_train, X_test, y_test)


# In[32]:

# Fine tuning the Gradient Boosting Classifier one param at a time
param_test1 = {u'n_estimators':[i for i in xrange(10, 1100, 10)]}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
                        min_samples_leaf=50,max_depth=8,max_features=u'sqrt',subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring=u'log_loss',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[33]:

param_test1 = {u'n_estimators':[i for i in xrange(90, 100)]}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,
                        min_samples_leaf=50,max_depth=8,max_features=u'sqrt',subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring=u'log_loss',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[34]:

param_test2 = {u'max_depth':[i for i in xrange(5,16,2)], u'min_samples_split': [i for i in xrange(200,1001,200)]}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_features=u'sqrt',
                        subsample=0.8,random_state=10,n_estimators=93), param_grid = param_test2, scoring=u'log_loss',
                        n_jobs=4,iid=False, cv=5)
gsearch2.fit(X_train, y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[35]:

param_test3 = {u'min_samples_leaf':[i for i in xrange(30,71,10)], u'min_samples_split': [i for i in xrange(10,200, 10)]}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_features=u'sqrt',max_depth=7,
                        subsample=0.8,random_state=10,n_estimators=93), param_grid = param_test3, scoring=u'log_loss',
                        n_jobs=4,iid=False, cv=5)
gsearch3.fit(X_train, y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[36]:

param_test5 = {u'max_features':[i for i in xrange(1,12,2)]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_depth=7,
                        min_samples_leaf= 40, min_samples_split= 170,subsample=0.8,random_state=10,n_estimators=93),
                        param_grid = param_test5, scoring=u'log_loss', n_jobs=4,iid=False, cv=5)
gsearch5.fit(X_train, y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_


# In[37]:

param_test6 = {u'subsample':[i/100 for i in xrange(1,100,5)]}
gsearch6 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=3,
                        min_samples_leaf= 40, min_samples_split= 170,random_state=10,n_estimators=93),
                        param_grid = param_test6, scoring=u'log_loss', n_jobs=4,iid=False, cv=5)
gsearch6.fit(X_train, y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[38]:

param_test6 = {u'subsample':[i/100 for i in xrange(90,101)]}
gsearch6 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=3,
                        min_samples_leaf= 40, min_samples_split= 170,random_state=10,n_estimators=93),
                        param_grid = param_test6, scoring=u'log_loss', n_jobs=4,iid=False, cv=5)
gsearch6.fit(X_train, y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[39]:

# Determining the best tradeoff for the number of estimators and learning rat
clf2 = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=3,
                        min_samples_leaf= 40, min_samples_split= 170,subsample=0.96,random_state=10,n_estimators=93)


# In[40]:

clf3 = GradientBoostingClassifier(learning_rate=0.05, max_depth=7, max_features=3,
                        min_samples_leaf= 40, min_samples_split= 170,subsample=0.96,random_state=10,n_estimators=186)


# In[41]:

clf4 = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features=3,
                        min_samples_leaf= 40, min_samples_split= 170,subsample=0.96,random_state=10,n_estimators=700)


# In[42]:

cross_val_score(clf2, X_all, y_all, cv = 3, scoring = u'log_loss')


# In[43]:

cross_val_score(clf3, X_all, y_all, cv = 3, scoring = u'log_loss')


# In[44]:

cross_val_score(clf4, X_all, y_all, cv = 3, scoring = u'log_loss')


# In[45]:

# Training best tuned model and preparing for submission to kaggle 
clf3.fit(X_all, y_all)
pred = clf3.predict_proba(x_test)


# In[47]:

submission = pd.DataFrame(pred, columns=clf3.classes_)


# In[48]:

submission[u'ID'] = testDF.ID


# In[49]:

submission.head()


# In[50]:

submission.to_csv(u'files/submission.csv', index=False)


# In[ ]:



