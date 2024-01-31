import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


data = pd.read_csv('creditcard.csv')
data.info()
data['Class'].value_counts()

data_majority = data[(data['Class'] == 0)]
data_minority = data[(data['Class'] == 1)]

data_minority_upsampled = resample(data_minority, replace=True, n_samples=282652, random_state=99)
data_final = pd.concat([data_minority_upsampled, data_majority])

sns.countplot(x= data_final['Class'])


data_final_x = data_final.drop(['Class'], axis=1)
data_final_y = data_final['Class']

x_train,x_test,y_train,y_test = train_test_split(data_final_x,data_final_y,test_size=0.1,random_state=99)

ss = StandardScaler()
x_train_f = ss.fit_transform(x_train)
x_test_f = ss.fit_transform(x_test)

pca = PCA()

x_train_f = pca.fit_transform(x_train_f)
x_test_f = pca.fit_transform(x_test_f)

print(f"x train shape: {x_train_f.shape}")
print(f"x test shape: {x_test_f.shape}")
print(f"y train shape: {y_train.shape}")
print(f"y test shape: {y_test.shape}")

model = XGBClassifier()
model.fit(x_train_f,y_train)
prediction = model.predict(x_test_f)

print(classification_report(y_test,prediction))
