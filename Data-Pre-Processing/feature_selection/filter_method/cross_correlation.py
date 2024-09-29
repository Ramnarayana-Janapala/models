import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


boston_house_price = pd.read_csv("../../../data/Boston.csv")
boston_house_price.head(5)
correlation = boston_house_price.corr()

plt.figure(figsize= (15,12))
sns.heatmap(correlation, annot=True)
plt.show()
