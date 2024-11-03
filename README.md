### Data Preprocessing Steps (README.md Explanation)

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

6. **Data Splitting**:
   - The dataset will be split into training, validation, and test sets to evaluate model performance on unseen data effectively.

---

### Instructions for Jupyter Notebook

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
   - For detailed data preprocessing and exploratory steps, refer to our main [Jupyter Notebook (Milestone2)](Sephora/cse151a_milestone2.py), which includes:
     - Step-by-step data exploration and preprocessing.
     - Code to handle missing values, encoding, and standardization.
     - Initial visualizations and insights into the dataset.
