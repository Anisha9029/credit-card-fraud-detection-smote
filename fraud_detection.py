# Credit Card Fraud Detection Project

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
df = pd.read_csv("credit_card_fraud_dataset.csv")

print("First 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns)

# Step 3: Check Target Column
# Change 'Class' if your dataset has different fraud column name
target_column = "Class"

if target_column not in df.columns:
    print("\nTarget column not found. Please check column name.")
    exit()

print("\nClass Distribution:")
print(df[target_column].value_counts())

# Step 4: Separate Features and Target
X = df.drop(target_column, axis=1)
y = df[target_column]

# Step 5: Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 7: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
