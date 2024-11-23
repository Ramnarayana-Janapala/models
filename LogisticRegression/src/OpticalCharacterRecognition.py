from sklearn.datasets import fetch_openml
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
X, y = fetch_openml('mnist_784', version=1,
                    return_X_y=True, as_frame=False)

print("Shape of input : ", X.shape, "Shape of target : ", y.shape)


#Plot the some of the training data set samples
plt.figure()
for idx, image in enumerate(X[:3]):
    plt.subplot(1, 3, idx + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
    plt.title('Training: ' + y[idx], fontsize = 20)
    plt.show()


y = [int(i) for i in y] # targets are strings, so need to convert to # int

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=1/7,random_state=0)
print("training samples shape = ", X_train.shape)
print("testing samples shape = ", X_test.shape)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(y_train, bins=20, rwidth=0.9, color='#607c8e')
plt.title('Frequency of different classes - Training Set')
plt.show()

plt.subplot(1,2,2)
plt.hist(y_test, bins=20, rwidth=0.9, color='#607c8e')
plt.title('Frequency of different classes - Test set')
plt.show()

model = LogisticRegression(fit_intercept=True,
                        multi_class='auto',
                        penalty='l1', #lasso regression
                        solver='saga',
                        max_iter=100,
                        C=50,
                        verbose=2, # output progress
                        n_jobs=5, # parallelize over 5 processes
                        tol=0.01
                         )

print(model)

model.fit(X_train, y_train)
print("Training Accuracy = ", np.around(model.score(X_train,   y_train)*100,3), "%")
print("Testing Accuracy = ", np.around(model.score(X_test, y_test)*100, 3), "%")



pred_y_test = model.predict(X_test)

cm = metrics.confusion_matrix(y_true=y_test,
                         y_pred = pred_y_test,
                        labels = model.classes_)


plt.figure(figsize=(12,12))

sns.heatmap(cm, annot=True,
            linewidths=.5, square = True, cmap = 'Blues_r', fmt='0.4g');

plt.ylabel('Actual label')
plt.xlabel('Predicted label')

index = 0
misclassified_images = []

for label, predict in zip(y_test, pred_y_test):

    if label != predict:
        misclassified_images.append(index)

    index += 1

    if len(misclassified_images) == 10:
        break

print("Ten Indexes are : ", misclassified_images)

plt.figure(figsize=(10, 10))
plt.suptitle('Misclassifications');

for plot_index, bad_index in enumerate(misclassified_images):
    p = plt.subplot(4, 5, plot_index + 1)  # 4x5 plot

    p.imshow(X_test[bad_index].reshape(28, 28), cmap=plt.cm.gray,
             interpolation='bilinear')
    p.set_xticks(());
    p.set_yticks(())  # remove ticks

    p.set_title(f'Pred: {pred_y_test[bad_index]}, Actual: {y_test[bad_index]}')
plt.show()

coef = model.coef_.copy()
scale = np.abs(coef).max()

plt.figure(figsize=(13, 7))

for i in range(10):  # 0-9

    coef_plot = plt.subplot(2, 5, i + 1)  # 2x5 plot

    coef_plot.imshow(coef[i].reshape(28, 28),
                     cmap=plt.cm.RdBu,
                     vmin=-scale, vmax=scale,
                     interpolation='bilinear')

    coef_plot.set_xticks(());
    coef_plot.set_yticks(())  # remove ticks
    coef_plot.set_xlabel(f'Class {i}')


plt.suptitle('Coefficients for various classes')
plt.show()