import dataframe as dataframe
import pandas as pd
import  numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

dataset = pd.read_csv("./dataset_cluster.csv", delimiter=",", dtype=np.uint8)

dataframe = pd.DataFrame(data=dataset, index=None, columns=None, dtype=np.uint8, copy=False)

scaler = StandardScaler()

#error --- check numb column and way to write column
dataframe.rename(columns ={"1601" : "Label"}, inplace =True) # Renaming the last column as "Label"
scaler.fit(dataframe.drop("Label", axis=1)) # Fit the data to all columns except Labels
scaled_features = scaler.transform(dataframe.drop("Label", axis=1))
df_feat = pd.DataFrame(scaled_features, columns=dataframe.columns[:-1])

# change label

X = df_feat
y = dataframe["Label"]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ten cluster
# knn = KNeighborsClassifier(n_neighbors=10)
# knn.fit(X_train, y_train)
# pred = knn.predict(X_test)
#
# print(confusion_matrix(y_test, pred))
# print(classification_report(y_test, pred))

# # only one cluster
# knn2 = KNeighborsClassifier(n_neighbors=1) # Only 1 cluster
# knn2.fit(X_train, y_train)
# pred2 = knn2.predict(X_test)
# print(confusion_matrix(y_test, pred2))
# print(classification_report(y_test, pred2))

error_rate = []

for i in range(11, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(4, 4))
plt.plot(range(1, 10), error_rate, color="blue", linestyle='dashed', marker="o", markerfacecolor="red", markersize=10)
plt.title("Error rate vs K-Value")
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.show()