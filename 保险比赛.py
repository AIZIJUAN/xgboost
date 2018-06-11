# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:20:44 2018

@author: sh02060
"""
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss,auc,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.cross_validation import  train_test_split
import xgboost as xgb
train=pd.read_csv(r'C:\Users\sh002060\Desktop\Homesite Quote Conversion\train.csv')
test=pd.read_csv(r'C:\Users\sh002060\Desktop\Homesite Quote Conversion\test.csv')
train.head()
train=train.drop('QuoteNumber',axis=1)
test=test.drop('QuoteNumber',axis=1)
train.columns
train.ix[0,'Original_Quote_Date']
type(train.ix[0,'Original_Quote_Date'])



train['DATE']=pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train=train.drop('Original_Quote_Date',axis=1)
test['DATE']=pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test=test.drop('Original_Quote_Date',axis=1)


train['Year']=train['DATE'].apply(lambda x: x.year)
train['Month']=train['DATE'].apply(lambda x: x.month)
train['weekday']=train['DATE'].apply(lambda x: x.weekday())
train.head()
train=train.drop('DATE',axis=1)

test['Year']=test['DATE'].apply(lambda x: x.year)
test['Month']=test['DATE'].apply(lambda x: x.month)
test['weekday']=test['DATE'].apply(lambda x: x.weekday())
test.head()
test=test.drop('DATE',axis=1)


train.isnull().any().any()

train.ix[:,train.isnull().any()]


train=train.fillna(-999)
test=test.fillna(-999)


a=set(train.columns)
b=set(test.columns)
a.difference(b)


feature =list(train.columns[1:])
print(feature)

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl=preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values)+list(test[f].values))
        train[f]=lbl.transform(list(train[f].values))
        test[f]=lbl.transform(list(test[f].values))


train.head()



Y_train = train['QuoteConversion_Flag']
X_train = train.drop(['QuoteConversion_Flag'],axis=1)
# create classifier
clf_tree = ExtraTreesClassifier(n_estimators=580,max_features=np.shape(X_train)[1],criterion='entropy',min_samples_split=3,max_depth=30,min_samples_leaf=8)
clf_xgboost = xgb.XGBClassifier(n_estimators=25,nthread=1,max_depth=10,learning_rate=0.025,silent=True,subsample=0.8,colsample_bytree=0.8)
'''
#cross validation
print 'cross validation...'
scores_tree = cross_val_score(clf_tree,X_train_new,Y_train,cv=3,scoring='roc_auc')
print 'random tree classifier score is:%s' % scores_tree
scores_xgboost = cross_val_score(clf_xgboost,X_train_new,Y_train,cv=3,scoring='roc_auc')
print 'xgboost classifier score is:%s' % scores_xgboost
'''

#predict probability
print ('predict probability')
clf_tree.fit(X_train,Y_train)
pred = clf_tree.predict_proba(test)[:,1]


#xgboost
print ('training model...')
params = {"objective":"binary:logistic"}
T_train_xgb = xgb.DMatrix(X_train,label=Y_train,missing=-999.0)
x_test_xgb = xgb.DMatrix(test,missing=-999.0)
gbm = xgb.train(params,T_train_xgb,20)
Y_pred = gbm.predict(x_test_xgb)
print ('finish training model...')
'''

#create submission
print ('create submission...')
submission =pd.DataFrame()
submission['QuoteNumber']=test['QuoteNumber']
submission['QuoteConversion_Flag']=pred
submission.to_csv('submission_tree_2015.1.16_3.csv',index=False)