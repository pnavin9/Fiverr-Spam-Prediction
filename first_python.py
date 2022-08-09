# Here are some basic libraries used and an overall program of the jupyter notebook
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = '../input/predict-potential-spammers-on-fiverr/train.csv'
df = pd.read_csv(path,index_col = 'user_id')
df.head()
from sklearn.model_selection import train_test_split
features = df.columns.drop('label')
X = df[features]
y = df['label']

X_train,X_val,y_train,y_val = train_test_split(X,y,random_state = 0) 
X.head()
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=500,learning_rate = 0.05)
model.fit(X_train,y_train,early_stopping_rounds = 5, eval_set = [(X_val,y_val)],verbose = False)
from sklearn.metrics import mean_absolute_error
pred = model.predict(X_val)
mean_absolute_error(y_val,pred)
path = '../input/predict-potential-spammers-on-fiverr/test.csv'
df = pd.read_csv(path,index_col = 'user_id')
df.head()
pred = model.predict(df)
df.columns
output = pd.DataFrame({'user_id': df.index,'prediction' : pred})
output.to_csv('submission.csv',index = False)
path = './submission.csv'
df = pd.read_csv(path,index_col = 'user_id')
df.head()
