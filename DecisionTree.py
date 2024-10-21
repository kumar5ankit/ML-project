# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\ak386\OneDrive\Desktop\python\project\cancerDataset.csv')

df = df.dropna()  
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Map 'M' and 'B' to 1 and 0

df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')  # Drop unnecessary columns

# Features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)  # Initialize the decision tree classifier
dt.fit(X_train, y_train)  # Train the model

# Predict on the test set
y_pred = dt.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Optional: Visualize feature importance for the Decision Tree
feature_importances = dt.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='skyblue')
plt.title('Feature Importances - Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()
