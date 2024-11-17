## Data Preprocessing Steps (README.md Explanation)

Our project aims to predict the popularity of Sephora products by analyzing metrics like product ratings, the number of reviews, and the "love" metric. To achieve this, we preprocess the data in the following ways:

1. **Handling Missing Values**:

   - We’ll check for any missing values in critical columns like `rating`, `reviews`, and `love` metrics. For numerical fields (e.g., `rating`, `price`), missing values will be imputed with the median or mean. For categorical fields (e.g., `brand`, `category`), we’ll consider using the mode or adding a category such as "Unknown."

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

We trained a **Linear Regression** model using the preprocessed data. Here are features used for prediction:
`brand`, `category`, `size`, `rating`, `number_of_reviews`, `love`, `MarketingFlags`, `online_only`, `exclusive`, `limited_edition`, `limited_time_offer`.
 The following steps were performed:

- **Data Split**: The data was split into a training set (80%) and a test set (20%).

- **Model Training**: A pipeline was built combining data preprocessing and model training using **Multiple Linear Regression**.


## Evaluation Metrics
1. The model was evaluated using Mean Squared Error (MSE) and R² Score for both training and test datasets. The following metrics were calculated:

   - Training MSE: 0.29784937212761403 (The model showed a very low error on the training set.)
   - Test MSE: 0.3694518357003383 (The error on the test set was significantly higher than the training set.)
   - Training R^2 Score: 0.704103802543203 (The score was very close to 1, suggesting that the model is fitting the training data well.)
   - Test R^2 Score: 0.6204882662063373 (The score was lower, indicating possible overfitting.)


2. **Question**: Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?
Based on the evaluation metrics, the Multiple Linear Regression model is likely overfitting the training data:

   - Training R² Score: 0.7041, indicating that the model captures a good portion of the variance in the training data.
   - Test R² Score: 0.6205, which is significantly lower than the training score, suggesting overfitting.
   - Training MSE: 0.2978, which is much lower compared to the test MSE.
   - Test MSE: 0.3695, indicating a gap between training and test performance.

   The discrepancy between the training and test scores implies that the model is learning details specific to the training data, such as noise, which reduces its ability to generalize to unseen data.

   To improve the model's generalization capabilities and address overfitting, we plan to explore regularization techniques such as **Ridge and Lasso Regression** will be applied to   penalize large coefficients, thereby preventing overfitting and encouraging simpler models that generalize better.

   Polynomial Regression: Considering the non-linear pattern observed in the residual plot, incorporating a polynomial regression model may help capture more complex relationships between the features and the target variable.

   Tree-based Models (Decision Tree): Since our current model might be missing non-linear relationships, using a tree-based model can provide better performance in capturing complex patterns without making strong parametric assumptions.

   Support Vector Machines (SVMs): For a more refined approach, SVMs with different kernel functions could also be explored. Like price and satisfied is non-linear and complex, an SVM with an RBF kernel could be a good choice. This would allow the model to draw non-linear boundaries in the feature space to better classify satisfied outcomes.

3. **Graphical Analysis**
   The following plots were used to analyze model performance:

   - **Residual Plot**: Visualized the residuals (errors) to identify any patterns. Ideally, the residuals should be randomly scattered around zero, which means the model has done a good job of capturing the data. Our plot showed almost no pattern, but there was a slight linear trend, suggesting that while the model is generally performing well, there may still be a small aspect of the data's complexity that it isn't fully capturing
     
   - **Distribution of Residuals**: Plotted the residuals' distribution to check if they were normally distributed. A normal distribution of residuals indicates that the model has no major biases and that the errors are evenly distributed. Deviations from normality may suggest model misspecification or the presence of outliers. In our case, the residuals looked roughly normal but not perfectly normal, which suggests that while the model performs reasonably well, there may still be minor biases or unaddressed complexities in the data.

   - **Predicted vs. Actual Plot**: Compared predicted values against actual values to evaluate how well the model captures the target variable. Ideally, the points should align closely with the line , which would indicate perfect predictions. Deviations from this line indicate areas where the model may be underpredicting or overpredicting. In our model, the points deviated from this line, indicating that the model's predictions were not always accurate, and there is room for improvement.

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
