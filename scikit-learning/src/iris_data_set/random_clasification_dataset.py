from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x,y = datasets.make_classification(n_samples=300, n_features=3, n_classes=3, n_redundant=0,n_clusters_per_class=1,weights=[0.5,0.3,0.2],random_state=42)
pca = PCA(n_components=2, svd_solver='randomized')
x_fitted = pca.fit_transform(x)
fit = pca.fit(x)

print((f"Explain Variance {fit.explained_variance_ratio_}"))

fig = plt.figure()

ax = fig.add_subplot(1,1,1,projection='3d')
ax.scatter(xs = x[:,0],ys=x[:,1],zs=x[:,2],c=y)
ax.set_title("Original 3-feature data")
ax.set_xlabel("x0")
ax.set_xlabel("x1")
ax.set_xlabel("x2")
plt.show()

fig,ax = plt.subplots(figsize=(9,6))
plt.title("Reduced 2-featured data")
plt.xlabel("x_fitted_0", fontsize=20)
plt.ylabel("x_fitted_1", fontsize=20)
plt.scatter(x_fitted[:,0],x_fitted[:,1],s=50,c=y)
plt.show()
