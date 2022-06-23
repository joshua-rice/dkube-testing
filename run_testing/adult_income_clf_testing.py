"""
Purpose: testing basic usage of DKube platform for a standard classification model task using the adult census income dataset

Creation date: 6/17/22
Created by: Josh Rice
"""

#### imports
import pandas as pd
import numpy as np
import category_encoders as ce

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import get_scorer, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from lightgbm import LGBMClassifier

from bayesian_model_tuner import BayesianModelTuner
from inspect import isclass

import mlflow
import joblib

import warnings
warnings.filterwarnings("ignore")




#### TODO: load data and do some basic feature engineering
data = pd.read_csv("/data/adult_census_income.csv")

train = data[data["test"]=="N"].copy(deep=True).reset_index(drop=True)
test = data[data["test"]=="Y"].copy(deep=True).reset_index(drop=True)

# encode response
train['target'] = 0
train.loc[train['income'] == '>50K', 'target'] = 1

test['target'] = 0
test.loc[test['income'] == '>50K', 'target'] = 1


# encode categorical vars
cat_cols = ['workclass', 'occupation', 'sex']
cat_cols_encode = []

con_cols = [
    'age', 
    'education.num', 
    'capital.gain', 
    'capital.loss', 
    'hours.per.week'
]

for col in cat_cols:
    le = LabelEncoder()
    train[col + '_int'] = le.fit_transform(train[col])
    test[col + '_int'] = le.transform(test[col])
    cat_cols_encode.append(col + '_int')

    
# define features and target
features = [
    'age', 
    'education.num', 
    'capital.gain', 
    'capital.loss', 
    'hours.per.week'
]

features += cat_cols_encode

target = 'target'



#### tuning and training (lgbm)
metadata = {
    "Executed by": "user",
    "Project": "examples",
    "Dataset": "Adult census income",
    "Date": "5/4/2042"
}

lgbm_search_params = {
    "n_estimators": scope.int(hp.quniform("n_estimators", 100, 2000, 50)),
    "learning_rate": hp.lognormal("learning_rate", -3, 1.5),
    "reg_lambda": hp.lognormal("reg_lambda", -1, 2),
    "min_child_weight": hp.lognormal("min_child_weight", -3, 1.5),
    "num_leaves": scope.int(hp.quniform("num_leaves", 10, 60, 2)),
    "max_depth": scope.int(hp.quniform("max_depth", 2, 10, 1)),
    "max_bin": scope.int(hp.quniform("max_bin", 20, 120, 5))
}

lgbm_fixed_params = {
    "random_state": 42,
    "boosting_type": "goss",
    "n_jobs": 16
}


## tuning
X = train[features]
y = train[target]

tuner = BayesianModelTuner(lgbm_search_params, LGBMClassifier, metadata, fixed_params=lgbm_fixed_params)
tuner.tune(X, y, 50, "roc_auc")
tuner.summary


## CV and train with tuned hyperparams
model = LGBMClassifier(**lgbm_fixed_params, **tuner.summary["best_params"])
cv_results = cross_val_score(model, X, y, cv=5)
mlflow.log_metric("Mean CV ROC-AUC", cv_results.mean())

model.fit(X,y)
train_pred_px = model.predict_proba(X)[:,1]
train_auc = roc_auc_score(y, train_pred_px)
mlflow.log_metric("Insample ROC-AUC", train_auc)



#### validate on test
test_pred_px = model.predict_proba(test[features])[:,1]
test_auc = roc_auc_score(test[target], test_pred_px)
mlflow.log_metric("Test ROC-AUC", test_auc)




#### export model
joblib.dump(model, "/model/model.joblib")