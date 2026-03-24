import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('income_random_forest.csv')

print(data.head())
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

target_col = 'Income'

X = data.drop(target_col, axis=1)
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_range = [10, 50, 100, 150, 200]
accuracies = []

for n in tree_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Trees: {n}, Accuracy: {acc:.4f}")

best_n = tree_range[np.argmax(accuracies)]
print("Best number of trees:", best_n)

rf_final = RandomForestClassifier(n_estimators=best_n, random_state=42)
rf_final.fit(X_train, y_train)


plt.figure()
plt.plot(tree_range, accuracies, marker='o')
plt.title('Accuracy vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.show()

importances = rf_final.feature_importances_
features = X.columns

indices = np.argsort(importances)

plt.figure()
plt.barh(features[indices], importances[indices])
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()