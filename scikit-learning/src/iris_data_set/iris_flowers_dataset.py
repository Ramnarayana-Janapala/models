import sklearn
from sklearn.datasets import load_iris
iris = load_iris()
x,y = iris.data,iris.target
features = iris.feature_names
labels = iris.target_names
print(f"Available Feature : {features}")
print(f"Available Categories : {labels}")
print(len(x))
print(len(y))
print(len(y[y==0]))
print(len(y[y==1]))
print(len(y[y==2]))
print(dir(sklearn.datasets))
# Filter out non-dataset functions
dataset_loaders = [d for d in dir(sklearn.datasets) if d.startswith('load_') or d.startswith('fetch_')]

print(dataset_loaders)