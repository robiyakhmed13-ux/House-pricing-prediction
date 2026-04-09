# Feature Engineering & Data Preprocessing for House Price Prediction

## Table of Contents
1. [Feature Engineering](#feature-engineering)
2. [Data Preprocessing](#data-preprocessing)
3. [Handling Missing Values](#handling-missing-values)
4. [Feature Scaling](#feature-scaling)
5. [Categorical Encoding](#categorical-encoding)
6. [Outlier Detection](#outlier-detection)
7. [Best Practices](#best-practices)

---

## Feature Engineering

### What is Feature Engineering?

Feature engineering is creating new features from existing ones to improve model performance.

### Why It Matters

```
Good Features       → Better Model
Bad/No Features    → Worse Model

Example:
Feature: 'square_footage' alone → R² = 0.75
Features: 'square_footage' + 'price_per_sqft' → R² = 0.85
```

### Common Feature Engineering Techniques

#### 1. Polynomial Features

```python
import numpy as np

# If larger house means exponentially higher price:
df['sqft_squared'] = df['square_footage'] ** 2
df['sqft_cubed'] = df['square_footage'] ** 3

# Example data:
# sqft: 1000, 2000, 3000
# sqft_squared: 1000000, 4000000, 9000000
```

#### 2. Interaction Features

```python
# Total rooms might matter more than bedrooms alone
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Price per bathroom
df['price_per_bath'] = df['price'] / df['bathrooms']

# House size relative to lot
df['sqft_ratio'] = df['square_footage'] / df['lot_size']

# Age as factor
df['decade_built'] = (2024 - df['year_built']) // 10
```

#### 3. Binning/Categorization

```python
# Create age categories
df['age_category'] = pd.cut(df['age'], 
    bins=[0, 5, 10, 20, 50, 100],
    labels=['Very New', 'New', 'Medium', 'Old', 'Very Old']
)

# Encode back to numeric if needed
df['age_category_numeric'] = pd.cut(df['age'], 
    bins=[0, 5, 10, 20, 50, 100]
).cat.codes
```

#### 4. Log Transformation

```python
# For skewed distributions (price often skewed)
df['price_log'] = np.log(df['price'])
df['sqft_log'] = np.log(df['square_footage'])

# Handles different scales better for some models
# Example: exponential relationship
```

#### 5. Domain-Specific Features

```python
# Location-based
df['is_downtown'] = (df['location'] == 'Downtown').astype(int)
df['distance_from_center'] = calculate_distance(df['lat'], df['lon'])

# Property-based
df['has_garage'] = (df['garage_spaces'] > 0).astype(int)
df['recent_renovation'] = (df['age'] < 10).astype(int)
df['luxury'] = ((df['bedrooms'] >= 4) & 
                (df['bathrooms'] >= 2.5)).astype(int)

# Market-based (if you have this data)
df['neighborhood_avg_price'] = df.groupby('location')['price'].transform('mean')
df['neighborhood_desirability'] = neighborhood_scores[df['location']]
```

#### 6. Temporal Features

```python
# If you have date data
df['month_sold'] = pd.to_datetime(df['sale_date']).dt.month
df['year_sold'] = pd.to_datetime(df['sale_date']).dt.year
df['quarter_sold'] = pd.to_datetime(df['sale_date']).dt.quarter

# Seasons
season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
              3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer',
              9: 'Fall', 10: 'Fall', 11: 'Fall'}
df['season_sold'] = df['month_sold'].map(season_map)
```

### Feature Selection

```python
# Correlation-based selection
correlations = df.corr()['price'].sort_values(ascending=False)
print(correlations)

# Keep features with |correlation| > 0.3
important_features = correlations[abs(correlations) > 0.3].index.tolist()

# For tree-based models
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)

importances = pd.Series(model.feature_importances_, 
                        index=X.columns).sort_values(ascending=False)
print(importances)

# Keep top N features
top_features = importances.head(10).index.tolist()
```

---

## Data Preprocessing

### Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
    
    def fit(self, df):
        """Fit preprocessor on training data"""
        # Identify column types
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fit encoders for categorical columns
        for col in self.categorical_cols:
            encoder = LabelEncoder()
            encoder.fit(df[col].dropna())
            self.encoders[col] = encoder
        
        # Fit scaler
        self.scaler.fit(df[self.numeric_cols])
    
    def transform(self, df):
        """Transform data"""
        df = df.copy()
        
        # Encode categorical
        for col in self.categorical_cols:
            if col in self.encoders:
                df[col] = self.encoders[col].transform(df[col])
        
        # Scale numeric
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        
        return df

# Usage
preprocessor = DataPreprocessor()
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

---

## Handling Missing Values

### Detection

```python
# Check for missing values
print(df.isnull().sum())

# Percentage
print(100 * df.isnull().sum() / len(df))

# Visualize
import matplotlib.pyplot as plt
df.isnull().sum().plot(kind='bar')
plt.title('Missing Values by Feature')
plt.show()
```

### Strategies

#### 1. Deletion
```python
# Drop rows with any missing values
df_clean = df.dropna()

# Drop rows where specific column is missing
df_clean = df.dropna(subset=['price'])

# Drop columns with >50% missing
df_clean = df.dropna(axis=1, thresh=0.5*len(df))
```

#### 2. Mean/Median Imputation
```python
# Fill numeric with mean
df['square_footage'].fillna(df['square_footage'].mean(), inplace=True)

# Fill with median (more robust to outliers)
df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)

# Fill specific columns
df.fillna({
    'bedrooms': df['bedrooms'].median(),
    'bathrooms': df['bathrooms'].median(),
    'age': df['age'].mean()
}, inplace=True)
```

#### 3. Forward Fill / Backward Fill
```python
# For time-series data
df['price'].fillna(method='ffill', inplace=True)  # Forward fill
df['price'].fillna(method='bfill', inplace=True)  # Backward fill
```

#### 4. Interpolation
```python
# Linear interpolation
df['price'].interpolate(method='linear', inplace=True)

# Polynomial interpolation
df['price'].interpolate(method='polynomial', order=2, inplace=True)
```

#### 5. Group-Based Imputation
```python
# Fill with group median
df['price'].fillna(df.groupby('location')['price'].transform('median'), 
                  inplace=True)

# Fill with group mean
df['age'].fillna(df.groupby('property_type')['age'].transform('mean'), 
                inplace=True)
```

#### 6. Advanced Methods (KNN Imputation)
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

### Decision Tree

```
Missing > 50%?
├─ YES → Remove column (too much missing)
└─ NO  → Is it numeric?
         ├─ YES → Fill with median
         └─ NO  → Fill with mode or drop
```

---

## Feature Scaling

### Why Scale?

```python
# Without scaling:
X = [[1000, 2],      # sqft, bedrooms
     [2000, 3],
     [3000, 4]]

# Features on different scales!
# SVR, KNN, Neural Networks are affected
# Tree-based models are NOT affected

# With scaling:
X_scaled = [[−1.22, −1.22],  # StandardScaler
            [0.00, 0.00],
            [1.22, 1.22]]

# Now on same scale!
```

### StandardScaler (Z-score normalization)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on training data
scaler.fit(X_train)

# Transform training and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Result: mean=0, std=1
print(X_train_scaled.mean(axis=0))  # ≈ 0
print(X_train_scaled.std(axis=0))   # ≈ 1
```

### MinMaxScaler (0-1 normalization)

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Result: all values in [0, 1]
print(X_train_scaled.min())  # 0
print(X_train_scaled.max())  # 1
```

### When to Scale

```
Scale Required (Do it!):
├─ SVR (Support Vector Regression)
├─ KNN (K-Nearest Neighbors)
├─ Neural Networks
├─ Logistic Regression
└─ Distance-based methods

Scale NOT Required:
├─ Decision Trees
├─ Random Forest
├─ Gradient Boosting
└─ Linear Regression (but helps)
```

---

## Categorical Encoding

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

# Fit and transform
df['location_encoded'] = encoder.fit_transform(df['location'])

# Result:
# Downtown → 0
# Suburbs → 1
# Rural → 2

# For prediction, use same encoder
new_location = encoder.transform(['Downtown'])  # → 0
```

### One-Hot Encoding

```python
# For categorical with few unique values
df_encoded = pd.get_dummies(df, columns=['location'])

# Result:
# location_Downtown: [1, 0, 0]
# location_Suburbs:  [0, 1, 0]
# location_Rural:    [0, 0, 1]

# Be careful with many categories!
# 100 categories → 100 new columns
```

### Ordinal Encoding

```python
# When categories have order
from sklearn.preprocessing import OrdinalEncoder

condition_map = {
    'Poor': 1,
    'Fair': 2,
    'Good': 3,
    'Excellent': 4
}

df['condition_encoded'] = df['condition'].map(condition_map)
```

---

## Outlier Detection

### Statistical Methods

```python
# Method 1: Standard Deviation
mean = df['price'].mean()
std = df['price'].std()

outliers = (df['price'] > mean + 3*std) | (df['price'] < mean - 3*std)
print(f"Outliers: {outliers.sum()}")

# Remove outliers
df_clean = df[~outliers]
```

```python
# Method 2: IQR (Interquartile Range)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

outliers = (df['price'] < Q1 - 1.5*IQR) | (df['price'] > Q3 + 1.5*IQR)
print(f"Outliers: {outliers.sum()}")

# Remove outliers
df_clean = df[~outliers]
```

```python
# Method 3: Isolation Forest
from sklearn.ensemble import IsolationForest

detector = IsolationForest(contamination=0.05)  # 5% outliers
outliers = detector.fit_predict(df[['price']]) == -1

print(f"Outliers: {outliers.sum()}")
df_clean = df[~outliers]
```

### Decision

```
Keep outliers if:
- They are valid data points
- You have reason to believe they are real
- They don't severely affect model

Remove outliers if:
- Data entry error (e.g., $0 price)
- Measurement error
- Significantly different (e.g., mansion vs normal house)
```

---

## Best Practices

### Complete Preprocessing Checklist

```
□ Load data
□ Check data types
□ Explore distributions
□ Identify missing values
□ Handle missing values (impute or remove)
□ Detect and handle outliers
□ Create new features (feature engineering)
□ Select important features
□ Encode categorical variables
□ Scale numeric features (if needed for model)
□ Split into train/test
□ IMPORTANT: Fit on train, transform test!
□ Train model
□ Evaluate
```

### Critical: Fit vs Transform

```python
# WRONG! Data leakage!
scaler = StandardScaler()
scaler.fit(X_total)  # Fit on all data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# RIGHT! No data leakage!
scaler = StandardScaler()
scaler.fit(X_train)  # Fit ONLY on train
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)  # Transform test with train's statistics

# Why? Test data should be "unseen" to model!
```

### Save Preprocessor

```python
import pickle

# Save after fitting
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Load for new predictions
with open('preprocessor.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Use for new data
new_data_scaled = scaler.transform(new_data)
```

---

**Happy Preprocessing! 🚀**
