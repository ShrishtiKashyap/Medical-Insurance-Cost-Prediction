# Medical Insurance Cost Prediction using Multiple Linear Regression

## Objective

The objective of this project is to develop a Multiple Linear Regression model to predict medical insurance charges using customer information such as age, sex, BMI, number of children, smoking status, and region. The project demonstrates the complete machine learning workflow, including data preprocessing, model development, prediction, and performance evaluation.

---

## Dataset

**Medical Cost Personal Insurance Dataset**

Dataset Source (Kaggle):  
https://www.kaggle.com/datasets/mirichoi0218/insurance

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

1. Loaded the dataset using Pandas.
2. Displayed the first five records and explored the dataset structure.
3. Identified numerical features, categorical features, and the target variable.
4. Checked the dataset for missing values.
5. Encoded the categorical variables (`sex`, `smoker`, and `region`) using LabelEncoder.
6. Split the dataset into 80% training data and 20% testing data.
7. Developed a Multiple Linear Regression model using the selected features.
8. Predicted insurance charges for the testing dataset.
9. Evaluated the model using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
10. Visualized the model performance using an Actual vs Predicted scatter plot.

---

## Results

The Multiple Linear Regression model was successfully trained and evaluated on the Medical Cost Personal Insurance dataset.

### Model Performance

- **Mean Absolute Error (MAE):** 4186.51
- **Mean Squared Error (MSE):** 33635210.43
- **R² Score:** 0.7833

### Key Findings

- The model explains approximately **78.33%** of the variation in medical insurance charges.
- Smoking status, BMI, and age have the strongest impact on insurance costs.
- The Actual vs Predicted scatter plot shows that the model predicts most insurance charges reasonably well, although prediction errors increase for some high-cost cases.

---

## Conclusion

The Multiple Linear Regression model successfully predicted medical insurance charges using demographic and health-related features. The evaluation metrics indicate that the model performs reasonably well, achieving an R² Score of 0.7833, which shows that the selected features explain a significant portion of the variation in insurance charges. Among all the input variables, smoking status, BMI, and age were the most influential factors affecting insurance costs. Although the model provides reliable baseline predictions, it assumes a linear relationship between the features and the target variable. More advanced machine learning algorithms such as Random Forest or Gradient Boosting can be explored in the future to improve prediction accuracy for complex, non-linear relationships.