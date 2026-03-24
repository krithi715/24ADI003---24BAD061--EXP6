# -----------------------------------
# Import Libraries
# -----------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------------
# Load Dataset
# -----------------------------------
df = pd.read_csv("fraud_smote.csv")   # update path if needed

print("First 5 rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

# -----------------------------------
# Check Class Imbalance
# -----------------------------------
print("\nClass Distribution BEFORE SMOTE:")
print(df['Fraud'].value_counts())

# Visualization BEFORE SMOTE
plt.figure()
df['Fraud'].value_counts().plot(kind='bar')
plt.title("Class Distribution BEFORE SMOTE")
plt.xlabel("Class (0=Normal, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# -----------------------------------
# Features and Target
# -----------------------------------
X = df.drop('Fraud', axis=1)
y = df['Fraud']

# -----------------------------------
# Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------
# Feature Scaling
# -----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# Model BEFORE SMOTE
# -----------------------------------
model_before = RandomForestClassifier()
model_before.fit(X_train, y_train)

y_pred_before = model_before.predict(X_test)
y_prob_before = model_before.predict_proba(X_test)[:, 1]

print("\nClassification Report BEFORE SMOTE:")
print(classification_report(y_test, y_pred_before))

# -----------------------------------
# Apply SMOTE
# -----------------------------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nClass Distribution AFTER SMOTE:")
print(pd.Series(y_train_sm).value_counts())

# Visualization AFTER SMOTE
plt.figure()
pd.Series(y_train_sm).value_counts().plot(kind='bar')
plt.title("Class Distribution AFTER SMOTE")
plt.xlabel("Class (0=Normal, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# -----------------------------------
# Model AFTER SMOTE
# -----------------------------------
model_after = RandomForestClassifier()
model_after.fit(X_train_sm, y_train_sm)

y_pred_after = model_after.predict(X_test)
y_prob_after = model_after.predict_proba(X_test)[:, 1]

print("\nClassification Report AFTER SMOTE:")
print(classification_report(y_test, y_pred_after))

# -----------------------------------
# Precision-Recall Curve
# -----------------------------------
precision_before, recall_before, _ = precision_recall_curve(y_test, y_prob_before)
precision_after, recall_after, _ = precision_recall_curve(y_test, y_prob_after)

pr_auc_before = auc(recall_before, precision_before)
pr_auc_after = auc(recall_after, precision_after)

# Print AUC values
print("\nPR AUC Before SMOTE:", pr_auc_before)
print("PR AUC After SMOTE:", pr_auc_after)

# Plot Precision-Recall Curve
plt.figure()

plt.plot(recall_before, precision_before, linestyle='--',
         label=f"Before SMOTE (AUC={pr_auc_before:.4f})")

plt.plot(recall_after, precision_after, linestyle='-',
         label=f"After SMOTE (AUC={pr_auc_after:.4f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Before vs After SMOTE)")
plt.legend()
plt.grid()
plt.show()