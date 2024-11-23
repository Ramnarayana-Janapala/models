import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')

# Loading dataset
df = pd.read_csv('./../data/apple_quality.csv')
print (df.head())
# Shape
print(df.shape)
# Information about dataset

print(df.dtypes)

# Typecasting acidity from object to float
def convert_acidity(acidity):
    try:
        if isinstance(acidity, float):
            result = acidity  # It's already a float
        else:
            acidity = str(acidity)  # Convert to string if it's not a float
            result = - float(acidity.replace('-', '')) if '-' in acidity else float(acidity)
        return result
    except ValueError:
        return np.nan


df['Acidity'] = df['Acidity'].apply(convert_acidity)
# Checking Null Data
print(df.isnull().sum())
# Dropping Null Values
df.dropna(inplace=True)

# Encoding Quality Feature
df['Quality'] = df['Quality'].map({"good":1, "bad":0})
print(df.head())

#Exploratory Data Analysis
# Distribution of Independent Features
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 8))
axs = axs.flatten()

features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness',
            'Acidity']  # Replace with actual feature names from your dataset

# Plot KDE for each numerical feature
for i, col in enumerate(features):
    sns.histplot(df[col], ax=axs[i], kde=True, color="#7701D1")
    axs[i].set_title(f'Distribution of {col}', fontsize=14)
    axs[i].set_xlabel(col, fontsize=12)
    axs[i].set_ylabel('Density', fontsize=12)

plt.suptitle("Distribution of Independent Features", size=25)
plt.subplots_adjust(hspace=0.5)

for j in range(len(features), len(axs)):
    axs[j].set_visible(False)

plt.show()
"""
All the features: Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness & Acidity — 
follow a normal distribution, with most values around the middle and fewer at the edges.
"""

# Dependent Feature ( Label DAta)
plt.figure(figsize=(6,3))
colors = sns.light_palette("#7701D1", n_colors=10)
sns.countplot(data=df, x='Quality', palette=[colors[4],colors[8]])
plt.xticks(ticks=[0,1], labels=['Bad', 'Good'])
plt.ylabel("Apple Count")
plt.title('Apply Quality Distribution')
plt.show()

"""
Both Good and Bad quality apples are equally proportionate 
"""

# See hoe each feature data is distributed
colors = sns.light_palette("#7701D1", n_colors=10)

plt.figure(figsize=(15, 10))
for i, column in enumerate(df.columns[1:-1]):
    plt.subplot(3, 3, i + 1)
    sns.boxenplot(x='Quality', y=column, data=df, palette=[colors[4], colors[8]])
    plt.xticks(ticks=[0, 1], labels=['Bad', 'Good'])
    plt.title(f'{column} by Quality')

plt.suptitle("Distribution of Apple Characteristics Across Good & Bad Quality", size=20)
plt.tight_layout()
plt.show()


"""
Bad-quality apples have lower Size, Sweetness, and Juiciness, but higher Crunchiness and Ripeness. 
* Good-quality apples show more balanced traits, with slightly better Juiciness and neutral values in other features, making them more desirable overall.
"""

# Correlation matrix
plt.figure(figsize=(10,5))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, annot=True, cmap='viridis', fmt='.2f', linewidths=2.5, cbar=True)
plt.title("Correlation Matrix", size=16)
plt.show()

"""
Influence on Apple Quality:
1. Size, Sweetness, and Juiciness positively correlate with better quality.
2. Ripeness negatively impacts quality.
3. Weight and Crunchiness have no significant effect on quality.
"""

# Outlier Detection
numerical_features = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']
df_melted = df[numerical_features].melt(var_name='Feature', value_name='Value')

plt.figure(figsize=(20, 5))
sns.boxplot(x='Feature', y='Value', data=df_melted, palette='Purples')
plt.title('Box Plot of Features with Outliers', size=18)
plt.xlabel('Feature')
plt.ylabel('Value')
plt.show()


"""
We can clearly see that the dataset contains a significant number of outliers.

"""

# Splitting Data into dependent & independent features
X = df.drop(columns=['Quality'], axis=1)
y = df['Quality']

# Train & Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", round(accuracy_score(y_test, y_pred) * 100), "%")

confusion_mat = confusion_matrix(y_test,y_pred,labels=None)

# Model Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)

y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', cbar=False,
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'], ax=axs[0])
axs[0].set_xlabel('Predicted Label')
axs[0].set_ylabel('True Label')
axs[0].set_title('Confusion Matrix')

# Plot ROC Curve
axs[1].plot(fpr, tpr, color='purple', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
axs[1].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
axs[1].set_xlim([0.0, 1.0])
axs[1].set_ylim([0.0, 1.05])
axs[1].set_xlabel('False Positive Rate')
axs[1].set_ylabel('True Positive Rate')
axs[1].set_title('Receiver Operating Characteristic (ROC)')
axs[1].legend(loc='lower right')

plt.suptitle("Model Evaluation", size=18)
plt.tight_layout()
plt.show()