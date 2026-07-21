# Medical Insurance Cost Prediction using Multiple Linear Regression

## Objective

The objective of this project is to build a Multiple Linear Regression model that predicts medical insurance charges based on customer information such as age, sex, BMI, number of children, smoking status, and region.

## Dataset

Medical Cost Personal Insurance Dataset

Kaggle Dataset:
https://www.kaggle.com/datasets/mirichoi0218/insurance

## Libraries Used

- pandas
- numpy
- matplotlib
- scikit-learn

## Methodology

1. Loaded the dataset using Pandas.
2. Displayed the first five records.
3. Identified numerical and categorical features.
4. Checked for missing values.
5. Encoded categorical variables using LabelEncoder.
6. Split the dataset into 80% training and 20% testing.
7. Built a Multiple Linear Regression model.
8. Predicted insurance charges.
9. Evaluated the model using MAE, MSE, and R² Score.
10. Visualized Actual vs Predicted values using a scatter plot.

## Results

The Multiple Linear Regression model was successfully trained and evaluated. The model produced reasonable predictions for insurance charges and achieved a good R² Score. The scatter plot showed that most predicted values were close to the actual values.

## Conclusion

The model demonstrated that smoking status, BMI, and age have a significant impact on medical insurance charges. Although Multiple Linear Regression provides good baseline performance, it assumes a linear relationship between features and the target variable. More advanced machine learning algorithms can further improve prediction accuracy.