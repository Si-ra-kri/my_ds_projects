# ======================
# Customer Churn Prediction
# ======================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load and explore data
print("Loading data...")
df = pd.read_csv("C:/Users/siva/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Quick data check
print("\nData sample:")
print(df.head())

print("\nChurn distribution:")
print(df['Churn'].value_counts())

# Simple visualization
plt.bar(['No', 'Yes'], df['Churn'].value_counts(), color=['green', 'red'])
plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")
plt.show()

# Preprocess data
print("\nPreprocessing data...")

# Convert target to binary
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert 'TotalCharges' to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Select features and make a copy to avoid SettingWithCopyWarning
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features].copy()
y = df['Churn']

# Fill missing numeric values with column means
X.fillna(X.mean(), inplace=True)

# Train-test split
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\nTraining model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
print("\nEvaluating model...")
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Interpret results
print("\nInterpretation...")
print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

# Simple insights
print("\nKey Insights:")
print("- Customers with longer tenure are less likely to churn (negative coefficient).")
print("- Higher monthly charges increase the likelihood of churn (positive coefficient).")
print("- Total charges also play a role but can be correlated with tenure.")
