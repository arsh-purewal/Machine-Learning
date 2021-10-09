import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

df.head()
df.info()
df.isnull().sum()
df.drop_duplicates(inplace=True)
df.info()
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(), annot = True)
from sklearn.model_selection import train_test_split
x = df[['time','serum_sodium','ejection_fraction','serum_creatinine']]
y = df.DEATH_EVENT
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state= 82)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_leaf_nodes=3, random_state=2, criterion='entropy')
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)
print(dtc.score(x_test, y_test))
