## Data Preprocessing Steps (README.md Explanation)

Our project aims to predict the popularity of Sephora products by analyzing metrics like product ratings, the number of reviews, and the "love" metric. To achieve this, we preprocess the data in the following ways:

1. **Handling Missing Values**:

   - We’ll check for any missing values in critical columns like `rating`, `reviews`, and `love` metrics. For numerical fields (e.g., `rating`, `price`), missing values will be imputed with the median or mean. For categorical fields (e.g., `brand`, `category`), we’ll consider using the mode or adding a category such as "Unknown."
   - After addressing missing values, we will explore the target variable (rating) to understand its distribution and behavior. Most products have ratings skewed toward higher values, indicating a general customer satisfaction trend. However, a few lower ratings suggest areas where specific products or categories may need improvement. Visualizing the rating distribution will help highlight these patterns and provide insight into potential anomalies or trends.

2. **Encoding Categorical Variables**:

   - Categorical fields like `brand` and `category` will be encoded. Non-ordinal fields will use one-hot encoding, creating binary columns for each category to ensure compatibility with machine learning models that require numerical inputs.
     - For example, if the `category` field contains values like "lipstick," "foundation," and "mascara," one-hot encoding will create three columns: `category_lipstick`, `category_foundation`, and `category_mascara`, with each row showing `1` in the respective column for its category and `0` in others.

3. **Normalization/Standardization of Numerical Features**:

   - Numerical fields like `price`, `rating`, and `love` metric values will be standardized or normalized. Standardization (mean = 0, std = 1) is particularly useful for regression models, as it ensures all numerical data contributes proportionally and improves model performance.

4. **Handling Outliers**:

   - Certain fields, like `price` or `number of reviews`, may contain outliers that could distort model performance. We will detect these outliers and consider capping or removing extreme values based on their distribution.

5. **Feature Engineering**:

   - Additional features, such as a `popularity_score` calculated from `rating`, `love`, and `number of reviews`, may be engineered to better capture a product’s overall appeal. We might also explore grouped features, like price ranges or aggregations for the `brand` and `category` fields.

6. **Dropped Unnecessary Columns**:

   - Columns such as `id`, `name`, `details`, `how_to_use`, `ingredients`, `options`, and `MarketingFlags_content` were removed as they did not contribute to predicting the target variable (`price`).

   - Target Variable: The target variable was set as `price`.

   - Identified Categorical and Numerical Features: Features were divided into numerical and categorical categories:

     - Numerical Features were scaled using `StandardScaler`.

     - Categorical Features were one-hot encoded using `OneHotEncoder`.

7. **Data Splitting**:
   - The dataset will be split into training, validation, and test sets to evaluate model performance on unseen data effectively.

## Model Training
### First Model

We trained a **Linear Regression** model using the preprocessed data. Here are features used for prediction:
`brand`, `category`, `size`, `rating`, `number_of_reviews`, `love`, `MarketingFlags`, `online_only`, `exclusive`, `limited_edition`, `limited_time_offer`.

To improve the model's generalization capabilities and address overfitting, we applied Ridge Regression to introduce a regularization term that penalizes large coefficients. 
The following steps were performed:

- **Data Split**: The data was split into a training set (80%) and a test set (20%).
- **Define hyperparameter grid for Ridge regression**
- **Model Training**: A pipeline was built combining data preprocessing and model training using **Multiple Linear Regression**.


### Evaluation Metrics

1. The model was evaluated using Mean Squared Error (MSE) and R² Score for both training and test datasets. The following metrics were calculated:

   - Training MSE: 0.2978493536286161 (The model showed a very low error on the training set.)
   - Test MSE: 0.3694577723488342 (The error on the test set was significantly higher than the training set.)
   - Training R^2 Score: 0.7041038209208921 (The score was very close to 1, suggesting that the model is fitting the training data well.)
   - Test R^2 Score: 0.620482167907328 (The score was lower, indicating possible overfitting.)

   In addition to evaluating the model using a single train-test split, we applied 5-fold cross-validation to understand the model's robustness and generalizability better. The results are as follows:

   - Cross-Validation MSE (Mean): 0.6404048139275803
   - Cross-Validation MSE (Standard Deviation): 0.1610573421083503
   
2. **Question**: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?
   Based on the evaluation metrics, the Multiple Linear Regression model is likely overfitting the training data:

   - Training R² Score: 0.7041, indicating that the model captures a good portion of the variance in the training data.
   - Test R² Score: 0.6205, which is significantly lower than the training score, suggesting overfitting.
   - Training MSE: 0.2978, which is much lower compared to the test MSE.
   - Test MSE: 0.3695, indicating a gap between training and test performance.
  
   - Test MSE (After Tuning): 0.3541 (The test error decreased, indicating improved generalization.)
   - Training MSE (After Tuning): 0.3092 (The training error increased slightly due to regularization.)
   - Training R^2 Score (After Tuning): 0.6928 (Slightly lower, indicating reduced complexity of the model.)
   - Test R^2 Score (After Tuning): 0.6362 (Improved compared to the original model, suggesting reduced overfitting.)

   The discrepancy between the training and test scores implies that the model is learning details specific to the training data, such as noise, which reduces its ability to generalize to unseen data.

   To improve the model's generalization capabilities and address overfitting, we plan to explore regularization techniques such as **Lasso Regression** will be applied to penalize large coefficients, thereby preventing overfitting and encouraging simpler models that generalize better.

   **Polynomial Regression**: Considering the non-linear pattern observed in the residual plot, incorporating a polynomial regression model may help capture more complex relationships between the features and the target variable.

   **Tree-based Models (Decision Tree)**: Since our current model might be missing non-linear relationships, using a tree-based model can provide better performance in capturing complex patterns without making strong parametric assumptions.

   **Support Vector Machines (SVMs)**: For a more refined approach, SVMs with different kernel functions could also be explored. Like price and satisfied is non-linear and complex, an SVM with an RBF kernel could be a good choice. This would allow the model to draw non-linear boundaries in the feature space to better classify satisfied outcomes.

3. **Graphical Analysis**
   The following plots were used to analyze model performance:

   - **Residual Plot**: Visualized the residuals (errors) to identify any patterns. Ideally, the residuals should be randomly scattered around zero, which means the model has done a good job of capturing the data. Our plot showed almost no pattern, but there was a slight linear trend, suggesting that while the model is generally performing well, there may still be a small aspect of the data's complexity that it isn't fully capturing
   - **Distribution of Residuals**: Plotted the residuals' distribution to check if they were normally distributed. A normal distribution of residuals indicates that the model has no major biases and that the errors are evenly distributed. Deviations from normality may suggest model misspecification or the presence of outliers. In our case, the residuals looked roughly normal but not perfectly normal, which suggests that while the model performs reasonably well, there may still be minor biases or unaddressed complexities in the data.

   - **Predicted vs. Actual Plot**: Compared predicted values against actual values to evaluate how well the model captures the target variable. Ideally, the points should align closely with the line , which would indicate perfect predictions. Deviations from this line indicate areas where the model may be underpredicting or overpredicting. In our model, the points deviated from this line, indicating that the model's predictions were not always accurate, and there is room for improvement.
  
4. **Conclusion - First Model**: After adding Ridge Regression, our model showed improved generalization capabilities compared to the initial Multiple Linear Regression model. Ridge regularization helped control overfitting by penalizing large coefficients, leading to a more balanced model performance across the training and test datasets. The training R² score (0.6927) was slightly reduced, but the test R² score improved to (0.6362), indicating better generalization. Additionally, the gap between the training MSE (0.3092) and the test MSE (0.3541) was reduced compared to the previous model, suggesting that the Ridge Regression model was better at capturing the complexity of the data without overfitting.

5. **Improment Plan**: While residuals were still close to normal, some slight patterns persisted, and the predicted vs. actual plot showed deviations, highlighting areas where the model could still be improved. To further enhance predictive accuracy, we plan to explore additional regularization techniques like Lasso Regression, which can help further reduce the influence of less important features. Additionally, we will consider non-linear models to capture more complex relationships and look into capping or removing extreme values to avoid skewing results.


### Second Model
For the second model, we trained a Decision Tree Regressor using the preprocessed data. Here are the main steps:
1. Hyperparameter Tuning: A Decision Tree model can be prone to overfitting, so we employed hyperparameter tuning to find the best model settings. The following hyperparameters were tuned using GridSearchCV:
   
   - `max_depth`: Limited tree depth to control model complexity.
   - `min_samples_split`: Minimum samples required to split an internal node.
   - `min_samples_leaf`: Minimum samples required to be at a leaf node.

2. Model Training:
   - A pipeline was built that combined data preprocessing and model training using a **Decision Tree Regressor**.
   - The hyperparameter tuning process was carried out with **5-fold** cross-validation to ensure robustness and prevent overfitting.

3. Evaluation Metrics
   The Decision Tree Regressor model was evaluated using the following metrics for both training and test datasets:
   - Training MSE: 0.1734 (The model showed low error on the training set.)
   - Test MSE: 0.4695 (The error on the test set was higher than the training set.)
   - Training R^2 Score: 0.8277 (Indicates a good fit for the training data.)
   - Test R^2 Score: 0.5178 (Lower compared to training, suggesting overfitting.)
The difference between training and test scores indicates that while the model fits the training data well, it struggles to generalize to new, unseen data. This discrepancy is a common issue for Decision Trees, especially when they are not properly regularized or when ensemble techniques are not used. The model may be learning specific details and noise in the training data, which prevents it from making accurate predictions on the test set.

4. Graphical Analysis:
   - Training vs. Test MSE: The bar chart comparing Training MSE and Test MSE highlights a significant gap, with the Training MSE being much lower. This suggests that the model has overfitted the training data, capturing noise and unnecessary complexity that does not generalize well to unseen data.
   - Residual Plot: The scatter plot of residuals shows increased variance for higher actual values. While some residuals are close to zero, others deviate significantly, indicating inconsistencies in the model’s ability to generalize, particularly for larger outputs. This pattern reflects potential overfitting or insufficient training data for those ranges.
   - Actual vs. Predicted Values: The scatter plot comparing actual and predicted values shows good alignment for smaller outputs, with many points near the diagonal line. However, noticeable deviations occur for higher actual values, revealing that the model struggles to predict larger outputs accurately, likely due to overfitting.

5. Question:
   - Where does your model fit in the fitting graph? From the Training vs. Test MSE Graph, we can see a significant gap between training and test MSE indicates overfitting. The model performs well on training data but generalizes poorly to unseen data. The growing spread of residuals as actual values increase(Residual Plot) suggests that the model struggles to capture the variance in the data, further confirming overfitting. While many predictions are close to the diagonal, deviations (especially for higher actual values) highlight the model's difficulty in accurately predicting for outliers or complex cases.
   - What are the next models you are thinking of and why? Gradient Boosting: Builds trees iteratively, correcting errors from previous trees. It often outperforms Random Forest in accuracy and offers strong control over bias-variance trade-offs with hyperparameter tuning

6. Predictions of correct: 411.
   
   FP: 718.

   FN: 705.

### Conclusion
#### First Model: Multiple Linear Regression
The Multiple Linear Regression model effectively captured a significant portion of the variance in the training data, as reflected by the Training R² Score (0.7041) and Training MSE (0.2978). However, its performance on the test set was weaker, with a Test R² Score (0.6205) and Test MSE (0.3695), indicating overfitting. The residual plot and predicted vs. actual plot revealed slight patterns and deviations, suggesting the model did not fully capture the data's complexity.

Improvement Plan: To address these issues, we plan to explore regularization techniques such as Ridge and Lasso Regression to prevent overfitting by penalizing large coefficients. Additionally, we will consider non-linear models to capture more complex relationships and handle extreme values to reduce their impact on the model.
#### Second Model: Decision Tree Regressor
The Decision Tree Regressor demonstrated strong performance on the training set, with a low Training MSE (0.1734) and a high Training R² Score (0.8277). However, the model exhibited significant overfitting, as indicated by the higher Test MSE (0.4695) and lower Test R² Score (0.5178). The residual plot showed increased variance for higher actual values, and the actual vs. predicted plot revealed notable deviations from the diagonal line for larger outputs. These results suggest that while the model fits the training data well, it struggles to generalize to unseen data, particularly for higher target values. In an effort to address the overfitting, we experimented with tuning hyperparameters such as 'model__max_depth': [7, 10, 15], which aimed to limit the tree’s complexity by restricting its depth. The updated model achieved a Training MSE of 0.4080 and Training R² Score of 0.5947, with a Test MSE of 0.5423 and Test R² Score of 0.4430. These results showed reduced overfitting, as evidenced by a smaller gap between training and test performance, but also demonstrated a trade-off in predictive accuracy.

Improvement Plan: Pruning the tree, further hyperparameter tuning, or adopting ensemble methods like Random Forest or Gradient Boosting could reduce overfitting. Additionally, expanding the dataset and enhancing feature engineering may help capture more complex patterns, particularly for higher-value predictions, thereby improving generalization and model robustness.

General: Both models demonstrated strengths but also exhibited clear signs of overfitting. Regularization, ensemble approaches, and better dataset balancing will be crucial in improving predictive performance and achieving more robust generalization across both approaches.

## Instructions for Jupyter Notebook

1. **Data Download and Environment Setup**:

   - At the top of the Jupyter Notebook, include commands to download and prepare the dataset:

     ```python
     # Install Kaggle library if not already installed
     !pip install kaggle

     # Download the dataset from Kaggle
     !kaggle datasets download -d raghadalharbi/all-products-available-on-sephora-website

     # Unzip the downloaded dataset
     !unzip all-products-available-on-sephora-website.zip
     ```

   - Also, ensure that necessary packages are installed for data processing and visualization:
     ```python
     !pip install pandas numpy scikit-learn matplotlib seaborn
     ```

2. **Link to Data**:

   - [Sephora dataset on Kaggle](https://www.kaggle.com/datasets/raghadalharbi/all-products-available-on-sephora-website)

3. **Link to the Jupyter Notebook**:
   - For detailed data preprocessing and exploratory steps, refer to our main [Jupyter Notebook (Milestone2)](./CSE151A_Milestone2.ipynb), which includes:
     - Step-by-step data exploration and preprocessing.
     - Code to handle missing values, encoding, and standardization.
     - Initial visualizations and insights into the dataset.
