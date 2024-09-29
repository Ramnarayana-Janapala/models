import pandas as pd
from mlxtend.feature_selection import ExhaustiveFeatureSelector as efs

### We will be using Linear Regression model to train on multiple combination of features.
from sklearn.linear_model import LinearRegression

boston_house_price = pd.read_csv("./../../data/Boston.csv")
result = boston_house_price.head(5)

#print(result)

features = boston_house_price.iloc[:,:13]
target = boston_house_price.iloc[:,-1]

### Creating the model with the required arguments of model and final number of features.

EFS = efs(estimator=LinearRegression(),
                                min_features=6,
                                max_features=12,
                                scoring = 'r2',
                                )

### Let's fit the above model on the features and target as supervised learning process.

EFS = EFS.fit(features, target)
print(f'Best fit indxes {EFS.best_idx_}')
print(f'Best fit Index (Corresponding Names) {EFS.best_feature_names_}')