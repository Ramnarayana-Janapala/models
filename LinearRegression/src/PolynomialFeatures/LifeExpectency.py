import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
 Predicating the Life Expectancy of a Person based on the country.
 Model Used : Liner Regression
 Used Forward filling for NaN values Replacement
 Used R^2 Score to evaluate the model
"""
def data_analysis(le):
    le_head = le.head(10)
    print(le.columns)
    le.sort_values(by='Income composition of resources', ascending=False).head(10)
    country_vs_life = le.groupby('Country', as_index=False)['Income composition of resources'].mean()
    country_vs_life.sort_values(by='Income composition of resources', ascending=False).head(10)
    print (country_vs_life)
    country_vs_life = le.groupby('Country', as_index=False)['Life expectancy '].mean()
    country_vs_life.sort_values(by='Life expectancy ', ascending=False).head(10)
    print(country_vs_life)
    plt.figure(figsize=(13, 8))
    sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1,
                  color_codes=True, rc=None)
    le_encoded = le.copy()
    # Replacing String with encoding values with labels, Convert Categorical Data to Numeric Data
    for column in le_encoded.select_dtypes(include=['object']).columns:
        le_encoded[column] = LabelEncoder().fit_transform(le_encoded[column])

    sns.histplot(le['Life expectancy'].dropna(), kde=True, color='blue')
    plt.show()
    # heat map for data co-relation
    cmap = sns.diverging_palette(500, 10, as_cmap=True)
    sns.heatmap(le_encoded.corr(), cmap=cmap, center=0, annot=False, square=True)

    plt.show()


def plot_nan_values(life):
    # Convert nan_counts Series to DataFrame for easier plotting
    nan_counts = life.isnull().sum()
    nan_counts_df = nan_counts.reset_index()
    nan_counts_df.columns = ['Feature', 'NaN Counts']

    # Plot the NaN counts using seaborn or matplotlib
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Feature', y='NaN Counts', data=nan_counts_df, palette='viridis')
    plt.title('NaN Counts by Feature')
    plt.xlabel('Features')
    plt.ylabel('NaN Counts')
    plt.xticks(rotation=45)
    plt.show()



def model_building(life):
    target = life['Life expectancy']
    features = life[life.columns.difference(['Life expectancy','Year'])]
    x_train,x_test,y_train,y_test = train_test_split(pd.get_dummies(features),target,test_size=0.2)
    model = LinearRegression()
    model.fit(x_train,y_train)
    y_predict = model.predict(x_train)
    residuals = y_train - y_predict
    lr_confidence = model.score(x_test, y_test)
    print('lr Confidence',lr_confidence)
    return y_predict, residuals

def plot_residuals(y_predict, residuals):
    plt.scatter(y_predict, residuals, color='blue', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')  # Reference line for zero residuals
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, linestyle='--', alpha=0.6)  # Optional: Add grid
    plt.show()

if __name__ == "__main__":
    le = pd.read_csv('./../../data/Life Expectancy Data.csv')
    #plot_nan_values(le)
    # fill nan values with forward data values
    le.ffill(inplace=True)

    #plot_nan_values(le)
    #data_analysis(le)
    y_redict , residuals = model_building(le)
    plot_residuals(y_predict=y_redict,residuals=residuals)