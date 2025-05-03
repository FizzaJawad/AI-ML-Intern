
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('creditcard.csv')

print("Class breakdown:")
print(df['Class'].value_counts())

features = df.drop(columns=['Class', 'Time'])
target = df['Class']

norm = StandardScaler()
features_scaled = norm.fit_transform(features)

oversample = SMOTE(random_state=0)
X_balanced, y_balanced = oversample.fit_resample(features_scaled, target)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_balanced, y_balanced, test_size=0.25, random_state=0)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train2, y_train2)

y_pred2 = lr_model.predict(X_test2)
y_prob2 = lr_model.predict_proba(X_test2)[:, 1]
print("\nConfusion Matrix:")
print(confusion_matrix(y_test2, y_pred2))
print("\nClassification Report:")
print(classification_report(y_test2, y_pred2))
print("ROC AUC Score:", roc_auc_score(y_test2, y_prob2))

sns.heatmap(confusion_matrix(y_test2, y_pred2), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()