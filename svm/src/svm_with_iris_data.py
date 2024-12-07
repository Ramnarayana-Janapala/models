import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score

iris_data = datasets.load_iris()
#print(iris_data)

#input features 4. sepal Width and sepal Length , petal Width and Petal Length
X = iris_data.data[:,]
Y = iris_data.target

iris_data_frame = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']], columns=iris_data['feature_names']+['target'])

plt.figure()
grr = pd.plotting.scatter_matrix(iris_data_frame, c=iris_data['target'],
                                  figsize=(15,5),
                                  s=60,alpha=0.8)
plt.show()

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

# Data Standardization
sc = StandardScaler()
sc.fit(x_train)
X_train_std = sc.transform(x_train)
X_test_std = sc.transform(x_test)

print(X_train_std)

model = SVC(kernel='rbf', random_state=0)
model.fit(x_train,y_train)

y_predit = model.predict(x_test)

# Evaluating model

matrix = confusion_matrix(y_test,y_predit)
fig, ax = plot_confusion_matrix(conf_mat=matrix, figsize=(6, 6), cmap=plt.cm.Greens)
plt.xlabel('Predictions', fontsize=15)
plt.ylabel('Actuals', fontsize=15)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

print(f"Accuracy Socre : {accuracy_score(y_test,y_predit)}")
