import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

data = pd.read_csv('churn_boosting.csv')
data = data.dropna(subset=['Churn'])
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna('Unknown')

le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

X = data.drop('Churn', axis=1)
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ada = AdaBoostClassifier(n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

y_prob_ada = ada.predict_proba(X_test)[:, 1]
y_prob_gb = gb.predict_proba(X_test)[:, 1]

fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_prob_gb)

auc_ada = auc(fpr_ada, tpr_ada)
auc_gb = auc(fpr_gb, tpr_gb)

plt.figure()
plt.plot(fpr_ada, tpr_ada, label=f'AdaBoost AUC = {auc_ada:.2f}')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting AUC = {auc_gb:.2f}')
plt.plot([0,1],[0,1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

importances = gb.feature_importances_
features = X.columns

indices = np.argsort(importances)

plt.figure()
plt.barh(features[indices], importances[indices])
plt.title('Feature Importance (Gradient Boosting)')
plt.xlabel('Importance')
plt.show()