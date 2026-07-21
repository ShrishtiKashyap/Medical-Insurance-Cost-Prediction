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
df.info()

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

predictions = pd.DataFrame({
    "Actual Charges": y_test.values,
    "Predicted Charges": y_pred
})

print("\nActual vs Predicted Charges:\n")
print(predictions.head(10))

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
print("1. Most predicted insurance charges are close to the actual values, indicating that the model performs reasonably well.")
print("2. The model predicts low and medium insurance charges more accurately than very high charges.")
print("3. Some deviation from the reference line indicates that Multiple Linear Regression cannot capture all complex relationships in the dataset.")

print("\n" + "=" * 60)
print("TASK 5: CONCLUSION")
print("=" * 60)

print("""
The Multiple Linear Regression model was successfully developed to predict medical insurance charges using customer information such as age, sex, BMI, number of children, smoking status, and region. After preprocessing the dataset and splitting it into training and testing sets, the model was trained and evaluated using MAE, MSE, and R² Score. The results showed that the model can reasonably estimate insurance charges. Smoking status, BMI, and age were the most influential factors affecting medical insurance costs, while region and sex had comparatively smaller effects. Although the model performs well, Multiple Linear Regression assumes a linear relationship between variables and may not accurately capture complex non-linear patterns. More advanced algorithms such as Random Forest or Gradient Boosting can improve prediction accuracy.
""")