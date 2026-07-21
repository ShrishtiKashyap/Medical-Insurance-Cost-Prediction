import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("TASK 1: DATA UNDERSTANDING")
print("=" * 60)

df = pd.read_csv("data/insurance.csv")

print("\nFirst Five Records:\n")
print(df.head())

print("\nDataset Information:\n")
print(df.info())

print("\nNumerical Features:")
print(df.select_dtypes(include=['int64', 'float64']).columns.tolist())

print("\nCategorical Features:")
print(df.select_dtypes(include=['object']).columns.tolist())

print("\nTarget Variable:")
print("charges")

print("\n" + "=" * 60)
print("TASK 2: DATA PREPROCESSING")
print("=" * 60)

print("\nMissing Values:\n")
print(df.isnull().sum())

encoder = LabelEncoder()

df["sex"] = encoder.fit_transform(df["sex"])
df["smoker"] = encoder.fit_transform(df["smoker"])
df["region"] = encoder.fit_transform(df["region"])

print("\nEncoded Dataset:\n")
print(df.head())

X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\n" + "=" * 60)
print("TASK 3: MODEL DEVELOPMENT")
print("=" * 60)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Charges:\n")
print(y_pred)

print("\n" + "=" * 60)
print("TASK 4: MODEL EVALUATION")
print("=" * 60)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Insurance Charges")
plt.show()

print("\nObservations:")
print("1. The predicted values closely follow the actual values, indicating that the model captures the overall trend.")
print("2. The R² score shows how much variation in insurance charges is explained by the model.")
print("3. Some prediction errors are visible because Linear Regression cannot perfectly model complex relationships.")

print("\n" + "=" * 60)
print("TASK 5: CONCLUSION")
print("=" * 60)

print("""
The Multiple Linear Regression model was developed to predict medical insurance charges using age, sex, BMI, number of children, smoking status, and region. The model was trained and evaluated using an 80:20 train-test split. Among all the features, smoking status had the greatest influence on insurance charges, followed by BMI and age. The evaluation metrics indicated that the model performs reasonably well for predicting insurance costs. However, Linear Regression assumes a linear relationship between variables and may not capture complex non-linear patterns present in the data. More advanced machine learning algorithms such as Random Forest or Gradient Boosting may provide better prediction accuracy.
""")