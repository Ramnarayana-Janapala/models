from setuptools.command.alias import alias
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


diabetes = pd.read_csv("./../../data/diabetes.csv")
target = diabetes["Outcome"]
features = diabetes.drop("Outcome",axis=1)

classifier = RandomForestClassifier(random_state=90, oob_score=True)
classifier.fit(features,target)

feature_importance = classifier.feature_importances_

feature_importance = 100.0 * (feature_importance/feature_importance.max())

sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]

pos = np.arange(sorted_idx.shape[0]) + 0.5

plt.figure(figsize=(10,12))
plt.barh(pos,feature_importance[sorted_idx],align='center')
plt.xticks(size=14)
plt.yticks(pos,features.columns[sorted_idx],size=14)
plt.xlabel('Relative importance', fontsize=15)
plt.title('Variable Importance', fontsize = 15)
plt.show()