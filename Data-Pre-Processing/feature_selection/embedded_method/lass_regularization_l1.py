from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

diabetes = pd.read_csv("./../../data/diabetes.csv")
target = diabetes["Outcome"]
features = diabetes.drop("Outcome",axis=1)

scaler = StandardScaler()
scaler.fit(features)
scaler_features = scaler.transform(features)

logistic = SelectFromModel(LogisticRegression(C=1, penalty='l1',solver='liblinear'))
logistic.fit(scaler_features,target)

selected_features = features.columns[(logistic.get_support())]

print(f'Total number of features : {features.shape[1]}')
print(f'Features Selected  : {len(selected_features)}')
print(f'Number of discarded features  : {np.sum(logistic.estimator_.coef_ == 0)}')

