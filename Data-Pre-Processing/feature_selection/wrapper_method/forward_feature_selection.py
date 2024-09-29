import pandas as pd
import numpy as n
### Python's mlxtend library provides a direct function for feature selection, which is SequentialFeatureSelector
from prettytable import PrettyTable
from mlxtend.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
### We will be using Linear Regression model to train on multiple combination of features.
from sklearn.linear_model import LinearRegression

boston_house_price = pd.read_csv("./../../data/Boston.csv")
result = boston_house_price.head(5)

#print(result)

features = boston_house_price.iloc[:,:13]
target = boston_house_price.iloc[:,-1]

### Creating the model with the required arguments of model and final number of features.

SFS = SequentialFeatureSelector(LinearRegression(),
                                k_features=4,
                                forward=False,
                                floating=False,
                                scoring = 'explained_variance',
                                cv=0)

### Let's fit the above model on the features and target as supervised learning process.

SFS.fit(features, target)
SFS_results = pd.DataFrame(SFS.subsets_).transpose()
# Create a PrettyTable object
#pd.options.display.float_format = '{:.2f}'.format
table = PrettyTable()

# Set the field names (column headers)
table.field_names = SFS_results.columns.tolist()

# Add rows from the DataFrame
for row in SFS_results.itertuples(index=False):
    table.add_row(row)
print(table)
avg_value = SFS_results.iloc[:,2]
plt.figure()
plt.plot(avg_value, color='green')
plt.xlabel("X")
plt.ylabel("avg_value")
plt.show()
