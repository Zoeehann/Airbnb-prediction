# Airbnb Listing Price Prediction — ML Pipeline

## Overview

This project builds a machine learning pipeline to predict Airbnb listing prices using a training dataset. The workflow includes data cleaning, feature engineering, exploratory data analysis, model training, and prediction.

The target variable is `log_price`, which represents the logarithm of the Airbnb listing price.

The implementation is written in Python using libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

---

## Project Structure

```
.
├── Han_WANG.ipynb        # Main notebook containing the full pipeline
├── airbnb_train.csv      # Training dataset
├── airbnb_test.csv       # Test dataset
└── README.md             # Project documentation
```

---

## Workflow

### 1. Data Loading

The project begins by loading the training and testing datasets using Pandas.

```python
df_train = pd.read_csv("airbnb_train.csv")
df_test  = pd.read_csv("airbnb_test.csv")
```

---

### 2. Data Cleaning

Several preprocessing steps are applied to improve data quality.

**Remove Duplicate Rows**  
Duplicate entries are removed from the training dataset.

**Handle Missing Values**  
Missing numerical values are filled using the median, which is robust to outliers.

Example features:
- `bathrooms`
- `bedrooms`
- `beds`

---

### 3. Feature Engineering

Additional features are created to improve model performance.

**Amenities Count**  
The number of amenities for each listing is calculated:

```
amenities_count = number of items in amenities list
```

**Boolean Mapping**  
String values such as `t` / `f` and `True` / `False` are converted into boolean values.

Example columns:
- `host_has_profile_pic`
- `host_identity_verified`
- `instant_bookable`

---

### 4. Data Validation

Invalid values are corrected or removed.

- Listings with `accommodates ≤ 0` are removed.
- Beds or bedrooms with invalid values are corrected.

Text fields are also standardized by:
- Removing extra spaces
- Converting to lowercase

---

### 5. Exploratory Data Analysis (EDA)

Several visualizations are used to understand the data distribution.

**Histogram of Numerical Variables**  
Helps observe the distribution of features.

**Correlation Heatmap**  
A heatmap is generated to examine correlations between numerical variables and identify relationships that may influence price prediction.

Libraries used: `Matplotlib`, `Seaborn`

---

### 6. Feature Encoding

Categorical variables are converted using One-Hot Encoding:

```python
pd.get_dummies()
```

This allows the model to process categorical data.

---

### 7. Model Training

The dataset is split into a **training set** and a **validation set**. The model predicts the target variable:

```
log_price
```

---

### 8. Model Evaluation

Two evaluation metrics are used:

| Metric | Description |
|--------|-------------|
| **RMSE** (Root Mean Squared Error) | Measures prediction error |
| **R² Score** | Indicates how well the model explains variance in the data |

Example output:

```
RMSE = ...
R²   = ...
```

A scatter plot of true vs. predicted prices is also generated to visualize model performance.

---

### 9. Prediction on Test Data

The trained model is used to generate predictions for the test dataset. Before prediction:
- Test features are encoded.
- Training and test features are aligned:

```python
X_encoded.align(df_test_encoded)
```

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Model training and evaluation |

---

## Key Skills Demonstrated

- Data preprocessing
- Feature engineering
- Exploratory data analysis
- Machine learning workflow
- Model evaluation
- Data visualization
