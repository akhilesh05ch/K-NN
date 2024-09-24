import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('User_Data.csv')
print(data)
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def calc_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def k_nearest_neighbors(train_features, train_labels, test_features, k):
    preds = []
    for test_instance in test_features:
        distances = [calc_distance(test_instance, train_instance) for train_instance in train_features]
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = [train_labels[i] for i in nearest_indices]
        most_common = Counter(nearest_labels).most_common(1)
        preds.append(most_common[0][0])
    return preds
def test_knn(k_list):
    accuracy_results = []
    for k in k_list:
        y_predictions = k_nearest_neighbors(X_train, y_train, X_test, k)
        acc = accuracy_score(y_test, y_predictions)
        accuracy_results.append(acc)
        print(f"K={k}, Accuracy={acc * 100:.2f}%")
    return accuracy_results
k_list = range(1, 21)
acc_list = test_knn(k_list)
plt.figure(figsize=(10, 6))
plt.plot(k_list, acc_list, marker='o')
plt.title("K vs Accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.xticks(k_list)
plt.grid(True)
plt.show()
optimal_k = k_list[np.argmax(acc_list)]
y_pred_optimal_k = k_nearest_neighbors(X_train, y_train, X_test, optimal_k)
conf_matrix = confusion_matrix(y_test, y_pred_optimal_k)
best_accuracy = accuracy_score(y_test, y_pred_optimal_k)
conf_matrix, best_accuracy, optimal_k
