import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data)
df['target'] = iris.target

x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), iris.target, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predicted = model.predict(x_test)
sns.set_theme(style="whitegrid")

cm = confusion_matrix(y_test,y_predicted)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
