# -*- coding: utf-8 -*-
"""CSE151A_Milestone4

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1t8b8URKDdLWWLJBvr55UtazFcEIFEXI8

## Import data
"""

import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# Download latest version
path = kagglehub.dataset_download("raghadalharbi/all-products-available-on-sephora-website")
files = os.listdir(path)
print("Path to dataset files:", path)
print("Files in dataset directory:", files)

csv_file_path = os.path.join(path, "sephora_website_dataset.csv")
data = pd.read_csv(csv_file_path)

"""## Data Exploration

***Column Descriptions***
- **id**: Unique identifier for each product.
- **brand**: Brand name of the product.
- **category**: Product category, e.g., skincare, fragrance.
- **name**: Name of the product.
- **size**: Product size information.
- **rating**: Customer rating of the product (scale 0–5).
- **number_of_reviews**: Number of customer reviews.
- **love**: "Love" metric, indicating user engagement.
- **price**: Product price in USD.
- **value_price**: Listed value price, if different from sale price.
- **MarketingFlags**: Boolean flag for marketing purposes.
- **MarketingFlags_content**: Additional marketing information.
- **options**: Available product options, such as colors or sizes.
- **details**: Detailed product description.
- **how_to_use**: Instructions for using the product.
- **ingredients**: List of ingredients for applicable products.
- **online_only**: Indicator if the product is only available online.
- **exclusive**: Indicator if the product is exclusive to Sephora.
- **limited_edition**: Flag for limited-edition items.
- **limited_time_offer**: Flag for limited-time offers.
"""

data.head()

print("\nData Info:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())

"""***Missing Data Summary***

The dataset has no missing values in any column, as confirmed by the analysis.

"""

print("\nDescriptive Statistics:")
data.describe()

count_name = len(pd.unique(data['name']))
print("Unique Products:", count_name)

count_category = len(pd.unique(data['category']))
print("Unique Categories:", count_category)

count_brand = len(pd.unique(data['brand']))
print("Unique Brands:", count_brand)

# Selecting numerical columns for standardization
numerical_features = ['price','number_of_reviews', 'love']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Extract relevant columns
rating = data['rating']
love = data['love']
reviews = data['number_of_reviews']

# Ensure no NaN or invalid values in 'reviews'
reviews = reviews.fillna(0)  # Replace NaN with 0
reviews = np.maximum(reviews, 1)  # Replace negatives or 0 with a minimum size of 1

# Calculate popularity score (scaled love metric)
scaler = MinMaxScaler()
popularity_score = scaler.fit_transform(love.values.reshape(-1, 1)).flatten()

# Create a scatter plot to visualize rating vs. popularity score
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    rating,
    popularity_score,
    c=love,
    s=[max(20, r * 5) for r in reviews],  # Adjusted size scaling
    cmap='viridis',
    alpha=0.5,  # Increase transparency
    edgecolor='k'
)
plt.colorbar(scatter, label='Love Metric (Scaled by Color)')
plt.xlabel('Rating')
plt.ylabel('Popularity Score')
plt.title('Relationship Between Rating and Popularity Score')
plt.grid(True)

# Add annotations only for points with high popularity scores
for i, txt in enumerate(reviews):
    if popularity_score[i] > 0.8:  # Annotate products with high popularity scores
        plt.annotate(
            int(txt),  # Display the review count as integer
            (rating.iloc[i], popularity_score[i]),
            fontsize=8,
            ha='right'
        )

plt.tight_layout()
plt.show()

"""### Target Variable Analysis: Rating
- Ratings are mostly positive, centered around higher values (e.g., 4 and above), indicating high customer satisfaction.
- Few low ratings suggest potential issues in specific product categories.
"""

# Analyze target variable (e.g., `rating`)
plt.figure(figsize=(10, 6))
sns.histplot(data['rating'], kde=True, bins=20, color='skyblue')
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Explanation
print("Ratings Summary:")
print(data['rating'].describe())

"""## Plot the Data and Visualization"""

# Histograms for continuous variables
numeric_columns = ['price', 'value_price', 'love', 'number_of_reviews', 'rating']
data[numeric_columns].hist(bins=15, figsize=(15, 10))
plt.suptitle('Histograms of Continuous Variables')
plt.show()



"""* Price: Most products have a price near zero, with very few high-priced items, indicating a heavily right-skewed distribution.
* Value Price: Similar to the price distribution, with a long right tail suggesting that a small number of items are much more expensive.
* Love: The distribution is heavily concentrated near the lower end, with most products having very low engagement (love count).
* Number of Reviews: The majority of products have few reviews, with the count sharply decreasing as the number of reviews increases.
* Rating: Ratings are concentrated around higher values, indicating that most products are well-rated.
"""

# Box plots for detecting outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[numeric_columns])
plt.title('Box Plot of Continuous Variables')
plt.show()

"""* Price & Value Price: There are significant outliers, especially in value_price, indicating that some products are exceptionally expensive compared to the majority.
* Love: A few products have significantly higher love counts, marking them as outliers.
* Number of Reviews: Most values are clustered near zero, with some extreme outliers suggesting a few products receive a disproportionately high number of reviews.
* Rating: Ratings have a relatively narrow range, with a few outliers.
"""

# Scatter plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='rating', data=data)
plt.title('Price vs Rating')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='number_of_reviews', data=data)
plt.title('Price vs Number of Reviews')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='love', data=data)
plt.title('Price vs Love')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='rating', y='love', data=data)
plt.title('Rating vs Love')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='rating', y='number_of_reviews', data=data)
plt.title('Rating vs Number of Reviews')
plt.show()

"""Observation:
1. Most products with low prices have fewer reviews, but there are a few expensive products with a high number of reviews. There is no obvious linear relationship.

2. Products with lower prices tend to have low love counts. However, some expensive products still receive high love counts, indicating mixed consumer engagement.

3. Higher-rated products are more likely to have a higher number of reviews, but there are exceptions. Some well-rated products still have few reviews.
"""

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data[numeric_columns].corr(), annot=True)
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()

"""
> price and value_price are highly correlated (0.98), suggesting redundancy. love and number_of_reviews also show a moderate positive correlation (0.75), indicating a potential relationship between these features.



"""

# Distribution plots for key columns
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(data['rating'], bins=10, kde=True, ax=axs[0]).set(title='Rating Distribution')
sns.histplot(data['price'], bins=10, kde=True, ax=axs[1]).set(title='Price Distribution')
sns.histplot(data['number_of_reviews'], bins=10, kde=True, ax=axs[2]).set(title='Number of Reviews Distribution')
plt.tight_layout()
plt.show()

"""* Rating: Ratings are mostly positive, centered around higher values, with a skew towards the upper end.
* Price: Prices are heavily skewed towards the lower end, with most products being affordable.
* Number of Reviews: The number of reviews is also skewed, with most products having few reviews and a small number having a significantly high count.

### Categorical Data Analysis
"""

plt.figure(figsize=(10, 6))
sns.countplot(y='category', data=data, order=data['category'].value_counts().nlargest(10).index)
plt.title('Top 10 Category')

top_brands = data['brand'].value_counts().nlargest(10)
plt.figure(figsize=(8, 6))
sns.barplot(y=top_brands.index, x=top_brands.values)
plt.title('Top 10 Brands')

"""# Preprocessing"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

"""### Major Preprocessing and Dropping unnecessary columns and handling categorical features, feature expansion"""

data_filtered = data.drop(columns=['id', 'name', 'details', 'how_to_use', 'ingredients', 'options', 'MarketingFlags_content'])

# Splitting the features and target variable
X = data_filtered.drop(columns=['price', 'value_price','URL'])
y = data_filtered['price']

# Identifying categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# Scale numerical features
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Define the polynomial degree
degree = 2

# Numerical preprocessing with scaling and polynomial features
numerical_preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Standardize numerical features
    ('poly', PolynomialFeatures(degree=degree, include_bias=False))  # Add polynomial features
])

# Preprocessing for categorical features
categorical_preprocessor = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_preprocessor, numerical_features),  # Use updated numerical preprocessor
        ('cat', categorical_preprocessor, categorical_features)
    ]
)

"""# Train first model"""

print("Features used for prediction:")
print(X.columns.tolist())

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

model = Ridge()

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Define hyperparameter grid for Ridge regression (regularization strength)
param_grid = {
    'model__alpha': [0.1, 1, 10, 50, 100]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Making predictions with the best model
y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)

# Showing predictions vs actual values after tuning
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
print("\nComparison of Actual vs Predicted Prices (After Hyperparameter Tuning):")
print(results.head(10))

# Evaluating the model after hyperparameter tuning
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Outputting the evaluation results after tuning
print("Training MSE (After Tuning):", train_mse)
print("Test MSE (After Tuning):", test_mse)
print("Training R^2 Score (After Tuning):", train_r2)
print("Test R^2 Score (After Tuning):", test_r2)

# Cross-validation for robustness
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mean_mse = -cv_scores.mean()
cv_std_mse = cv_scores.std()
print("\nCross-Validation MSE (Mean):", cv_mean_mse)
print("Cross-Validation MSE (Standard Deviation):", cv_std_mse)

# Draw the linear regression model graph
plt.figure(figsize=(10, 6))
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, color='dodgerblue', label='Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True)
plt.tight_layout()
plt.show()

"""

> The residual plot shows the difference between the actual values and the predicted values of our model. The residuals are scattered around the horizontal line at zero, which indicates that the errors have a roughly constant variance. However, the fan shape of the scatter indicates some heteroscedasticity, suggesting that our model may not perfectly fit the data, particularly as predicted values increase.

"""

# Update the log-transformed residual, to detect skewness

# Calculate residuals
residuals = y_test - y_test_pred

# Apply logarithmic transformation
log_residuals = np.log1p(np.abs(residuals))  # Use log(1 + |residual|)

# Plot raw residuals histogram
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Raw Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot log-transformed residuals histogram
plt.figure(figsize=(10, 6))
plt.hist(log_residuals, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Log of Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Log-Transformed Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()

"""> This histogram shows the frequency distribution of the residuals. The residuals appear to be centered around zero, which indicates that the model's predictions are unbiased on average. However, the spread and slight skewness of the residuals hint at potential issues with the model's assumptions about error distribution."""

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Price')
plt.grid(True)
plt.tight_layout()
plt.show()

"""> This scatter plot compares the predicted prices to the actual prices. The red dashed line represents a perfect prediction (where predicted values would exactly match actual values). The spread around the line indicates the error in predictions, and it is clear that our model performs well for lower actual prices but has higher error for larger prices, as points deviate more from the line at higher values.

# Question:

Where does your model fit in the fitting graph?

Based on the residual plots and the predicted vs. actual plot, it appears that our model is in the middle range between underfitting and overfitting. The model shows reasonable performance in capturing the trend of the data, as indicated by the relatively low variance of residuals and the reasonable clustering around the line in the predicted vs. actual plot.

However, the scatter of residuals and the deviation of data points from the red dashed line at higher prices indicate that our model may be over-simplifying the relationships between features and the target variable. This suggests that while the model is not underfitting, it may not be capturing all the complex patterns in the data.

What are the next models you are thinking of and why?

* Polynomial Regression: Considering the non-linear pattern observed in the residual plot, incorporating a polynomial regression model may help capture more complex relationships between the features and the target variable.
* Tree-based Models (Decision Tree): Since our current model might be missing non-linear relationships, using a tree-based model can provide better performance in capturing complex patterns without making strong parametric assumptions.
* Support Vector Machines (SVMs): For a more refined approach, SVMs with different kernel functions could also be explored. Like price and satisfied is non-linear and complex, an SVM with an RBF kernel could be a good choice. This would allow the model to draw non-linear boundaries in the feature space to better classify satisfied outcomes.

###Conclusion
After adding Ridge Regression, our model showed improved generalization capabilities compared to the initial Multiple Linear Regression model. Ridge regularization helped control overfitting by penalizing large coefficients, leading to a more balanced model performance across the training and test datasets. The training R² score (0.6927) was slightly reduced, but the test R² score improved to (0.6362), indicating better generalization. Additionally, the gap between the training MSE (0.3092) and the test MSE (0.3541) was reduced compared to the previous model, suggesting that the Ridge Regression model was better at capturing the complexity of the data without overfitting.

While residuals were still close to normal, some slight patterns persisted, and the predicted vs. actual plot showed deviations, highlighting areas where the model could still be improved. To further enhance predictive accuracy, we plan to explore additional regularization techniques like Lasso Regression, which can help further reduce the influence of less important features. Additionally, we will consider non-linear models to capture more complex relationships and look into capping or removing extreme values to avoid skewing results.

# Train the second Model (Decision Tree)
"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Decision Tree
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor())
])

# Define hyperparameters to tune
param_grid = {
    'model__max_depth': [3, 5, 7, 10, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

"""## Hyperparameter Tunning"""

# Perform hyperparameter tuning with GridSearchCV
dt_grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
dt_grid_search.fit(X_train, y_train)

# Get the best model
best_dt_model = dt_grid_search.best_estimator_

y_test_pred = best_dt_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Best Parameters:", dt_grid_search.best_params_)
print("Test MSE:", test_mse)
print("Test R^2 Score:", test_r2)

# Evaluate the model on training data
y_train_pred = best_dt_model.predict(X_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("Training MSE:", train_mse)
print("Training R^2 Score:", train_r2)

"""## Visualization"""

# Plotting Training vs Test MSE
plt.figure(figsize=(10, 6))
plt.bar(['Training MSE', 'Test MSE'], [train_mse, test_mse], color=['grey', 'lightskyblue'])
plt.ylabel('Mean Squared Error')
plt.title('Training vs Test MSE')
plt.show()

"""The **training MSE**(Gray Bar)is relatively low, indicating that the Decision Tree model has learned the training data very well.

The **test MSE** is significantly higher than the training MSE. This gap suggests that the model is **overfitting** the training data, meaning it has captured noise or unnecessary complexity in the training data that does not generalize well to unseen data.
"""

# Plotting Residuals
residuals = y_test - y_test_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.5,color='mediumseagreen')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

"""The residuals are scattered around the horizontal axis (0 residual line), but there is visible variance that increases as the actual values grow larger. While some residuals are close to zero (indicating good predictions), others deviate significantly, suggesting inconsistencies in the model's ability to generalize across the dataset. This pattern suggests the model struggles with higher actual values, possibly due to overfitting or insufficient training data for those ranges."""

# Plotting Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='sandybrown')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

"""The **Actual vs Predicted Values** plot shows how closely the model's predictions align with the true values. While many points are near the diagonal (perfect predictions), noticeable deviations, especially for higher actual values, suggest the model struggles with accurate predictions for larger outputs, likely due to overfitting or limited generalization."""

import numpy as np

# Define a tolerance range (e.g., 10% of the actual value)
tolerance = 0.1 * np.abs(y_test)  # 10% of actual value

# Identify correct, FP, and FN predictions
correct_predictions = ((y_test_pred >= (y_test - tolerance)) & (y_test_pred <= (y_test + tolerance))).sum()
false_positives = (y_test_pred > (y_test + tolerance)).sum()  # Overestimated
false_negatives = (y_test_pred < (y_test - tolerance)).sum()  # Underestimated

# Print the results
print("Correct Predictions:", correct_predictions)
print("False Positives (FP):", false_positives)
print("False Negatives (FN):", false_negatives)

"""## **Questions**:

**Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?**


*   Where does your model fit in the fitting graph?

  Where does your model fit in the fitting graph?
  From the Training vs. Test MSE Graph, we can see a significant gap between training and test MSE indicates overfitting. The model performs well on training data but generalizes poorly to unseen data.

  The growing spread of residuals as actual values increase(Residual Plot) suggests that the model struggles to capture the variance in the data, further confirming overfitting.

  While many predictions are close to the diagonal, deviations (especially for higher actual values) highlight the model's difficulty in accurately predicting for outliers or complex cases.



*   What are the next models you are thinking of and why?

##  **Conclusion section**

What is the conclusion of your 2nd model? What can be done to possibly improve it? Note: The conclusion section should be it's own independent section. i.e. Methods: will have models 1 and 2 methods, Conclusion: will have models 1 and 2 results and discussion.

**First Model: Multiple Linear Regression**

*   The Multiple Linear Regression model effectively captured a significant portion of the variance in the training data, as reflected by the Training R² Score (0.7041) and Training MSE (0.2978). However, its performance on the test set was weaker, with a Test R² Score (0.6205) and Test MSE (0.3695), indicating overfitting. The residual plot and predicted vs. actual plot revealed slight patterns and deviations, suggesting the model did not fully capture the data's complexity.
*   **Improvement Plan:**
To address these issues, we plan to explore regularization techniques such as Ridge and Lasso Regression to prevent overfitting by penalizing large coefficients. Additionally, we will consider non-linear models to capture more complex relationships and handle extreme values to reduce their impact on the model.

**Second Model: Decision Tree Regressor**

*   The Decision Tree Regressor demonstrated strong performance on the training set, with a low Training MSE (0.1734) and a high Training R² Score (0.8277). However, the model exhibited significant overfitting, as indicated by the higher Test MSE (0.4695) and lower Test R² Score (0.5178). The residual plot showed increased variance for higher actual values, and the actual vs. predicted plot revealed notable deviations from the diagonal line for larger outputs. These results suggest that while the model fits the training data well, it struggles to generalize to unseen data, particularly for higher target values. In an effort to address the overfitting, we experimented with tuning hyperparameters such as 'model__max_depth': [7, 10, 15], which aimed to limit the tree’s complexity by restricting its depth. The updated model achieved a Training MSE of 0.4080 and Training R² Score of 0.5947, with a Test MSE of 0.5423 and Test R² Score of 0.4430. These results showed reduced overfitting, as evidenced by a smaller gap between training and test performance, but also demonstrated a trade-off in predictive accuracy.
*   **Improvement Plan:** Pruning the tree, further hyperparameter tuning, or adopting ensemble methods like Random Forest or Gradient Boosting could reduce overfitting. Additionally, expanding the dataset and enhancing feature engineering may help capture more complex patterns, particularly for higher-value predictions, thereby improving generalization and model robustness.

**General:** Both models demonstrated strengths but also exhibited clear signs of overfitting. Regularization, ensemble approaches, and better dataset balancing will be crucial in improving predictive performance and achieving more robust generalization across both approaches.
"""