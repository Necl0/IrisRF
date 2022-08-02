import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


digits = load_digits()

df = pd.DataFrame(digits.data)
df['target'] = digits.target
df.head()

x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), digits.target, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

model.score(x_test, y_test)
