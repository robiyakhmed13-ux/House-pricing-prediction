# Machine Learning Regression Models - Detailed Explanation

## Table of Contents
1. [Regression vs Classification](#regression-vs-classification)
2. [Linear Regression](#linear-regression)
3. [Decision Tree Regression](#decision-tree-regression)
4. [Random Forest Regression](#random-forest-regression)
5. [Gradient Boosting Regression](#gradient-boosting-regression)
6. [Support Vector Regression](#support-vector-regression)
7. [Model Comparison](#model-comparison)
8. [Evaluation Metrics](#evaluation-metrics)

---

## Regression vs Classification

### Key Differences

```
REGRESSION:
  Output: Continuous numeric value
  Example: Predict house price = $450,000
  Loss Function: MSE, MAE, RMSE
  Use Cases: Price prediction, temperature, stock price
  
CLASSIFICATION:
  Output: Discrete category/class
  Example: Predict price category = "Expensive"
  Loss Function: Cross-entropy, accuracy
  Use Cases: Spam detection, disease diagnosis
```

### Why Regression for House Prices?

House prices are continuous values (e.g., $345,000.50, $450,000, $1,200,000).
Regression naturally handles this continuous output space, unlike classification
which would force prices into discrete buckets (expensive/medium/cheap).

---

## Linear Regression

### Concept

Linear regression finds the best-fitting straight line through data points.

**Mathematical Model**:
```
y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ

Where:
- y: Predicted price
- x₁, x₂, ..., xₙ: Features (square footage, bedrooms, etc.)
- b₀: Intercept (base price)
- b₁, b₂, ..., bₙ: Coefficients (price impact of each feature)
```

### Visual Representation

```
Price
  ^
  |     *
  |  *     *
  | *  *     *
  |*    *      *
  |--*----*------*--> Square Footage
  |        *   *
  |         * *
  |
  └────────────────
  
Regression line finds the best fit
```

### Advantages
- ✓ Simple and interpretable
- ✓ Fast training and prediction
- ✓ Works well with linear relationships
- ✓ Requires fewer samples
- ✓ Good baseline model

### Disadvantages
- ✗ Cannot capture non-linear patterns
- ✗ Assumes linear relationship between features and price
- ✗ Sensitive to outliers
- ✗ Performance limited by linearity assumption

### Example Output

```python
model = LinearRegression()
model.fit(X_train, y_train)

# Get coefficients
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: ${coef:,.2f} per unit")
    
# Example output:
# square_footage: $150.00 per sq ft
# bedrooms: $50,000.00 per bedroom
# bathrooms: $30,000.00 per bathroom
```

---

## Decision Tree Regression

### Concept

A decision tree makes hierarchical decisions to predict prices.

**Decision Process**:
```
Is square_footage > 2000?
├─ YES:
│  ├─ Are there > 3 bedrooms?
│  │  ├─ YES: Predict price = $450,000
│  │  └─ NO: Predict price = $350,000
│  └─ Are bathrooms > 2?
│     ├─ YES: Predict price = $400,000
│     └─ NO: Predict price = $300,000
└─ NO:
   ├─ Is age < 10 years?
   │  ├─ YES: Predict price = $250,000
   │  └─ NO: Predict price = $200,000
```

### How It Works

1. Split data on feature that reduces error most
2. Recursively split resulting subsets
3. Continue until reaching leaf nodes (predictions)
4. For regression: leaf value = mean of samples in that leaf

### Advantages
- ✓ Captures non-linear relationships
- ✓ Easy to visualize and interpret
- ✓ No feature scaling needed
- ✓ Handles both numeric and categorical data
- ✓ Feature importance available

### Disadvantages
- ✗ Prone to overfitting
- ✗ Small data changes cause large tree changes
- ✗ Can create overly complex trees
- ✗ Biased toward dominant features

### Controlling Overfitting

```python
# Too deep (overfitting)
model = DecisionTreeRegressor(max_depth=None)

# Moderate depth (good balance)
model = DecisionTreeRegressor(max_depth=10)

# Shallow tree (underfitting)
model = DecisionTreeRegressor(max_depth=3)
```

---

## Random Forest Regression

### Concept

Random Forest combines multiple decision trees (ensemble method).

**Process**:
```
1. Create many random subsets of data
2. Train a decision tree on each subset
3. Each tree makes a prediction
4. Final prediction = average of all tree predictions

Tree 1:  $450,000
Tree 2:  $460,000
Tree 3:  $455,000
Tree 4:  $440,000
Tree 5:  $452,000
────────────────
Average: $451,400 (final prediction)
```

### Why It Works Better

- **Bagging**: Uses random samples → reduces variance
- **Randomness**: Each tree sees different data → diverse predictions
- **Averaging**: Combines predictions → smooths errors

### Advantages
- ✓ Better generalization than single tree
- ✓ Reduces overfitting significantly
- ✓ Handles non-linear relationships
- ✓ Robust to outliers
- ✓ Feature importance ranking
- ✓ Parallelizable (fast training)

### Disadvantages
- ✗ Less interpretable than single tree
- ✗ More memory required
- ✗ Training slower than single tree
- ✗ Can overfit with high tree depth

### Key Parameters

```python
model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=15,          # Max tree depth
    min_samples_split=2,   # Min samples to split node
    min_samples_leaf=1,    # Min samples in leaf
    random_state=42        # Reproducibility
)
```

---

## Gradient Boosting Regression

### Concept

Gradient Boosting builds trees sequentially, each correcting previous errors.

**Process**:
```
Step 1: Train Tree 1 on all data
        Errors: [+$10k, -$5k, +$8k, -$3k, ...]
        
Step 2: Train Tree 2 to predict the errors from Tree 1
        Helps correct mistakes
        
Step 3: Train Tree 3 to predict remaining errors from Trees 1+2
        Further improvement
        
Step 4-100: Continue building trees, each improving on previous

Final Prediction = Tree 1 + Tree 2 + Tree 3 + ... + Tree 100
```

### Mathematical View

```
y_pred = y_pred₀ + lr × tree₁_pred + lr × tree₂_pred + ...

Where:
- y_pred₀: Initial prediction
- lr: Learning rate (typically 0.01-0.1)
- treeᵢ_pred: Prediction from tree i (predicts residuals)
```

### Advantages
- ✓ Often achieves best performance
- ✓ Reduces both bias and variance
- ✓ Handles non-linear relationships well
- ✓ Feature importance available
- ✓ Good default algorithm to try

### Disadvantages
- ✗ Complex hyperparameters
- ✗ Slower training than Random Forest
- ✗ More prone to overfitting if not careful
- ✗ Requires careful tuning
- ✗ Less interpretable

### Key Parameters

```python
model = GradientBoostingRegressor(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinks contributions
    max_depth=5,             # Max tree depth
    subsample=1.0,           # Fraction of samples per tree
    random_state=42
)

# Lower learning rate → slower learning, may need more estimators
# Higher learning rate → faster learning, risk of overfitting
```

### Hyperparameter Tuning Impact

```
Small learning_rate (0.01):
  - Pro: More stable, less overfitting
  - Con: Needs more trees, slower training
  
Large learning_rate (0.1):
  - Pro: Fewer trees needed, faster
  - Con: May overshoot, higher variance
  
Deep trees (max_depth=10):
  - Pro: Captures complex patterns
  - Con: Overfitting risk
  
Shallow trees (max_depth=3):
  - Pro: Simple, less overfitting
  - Con: May underfit
```

---

## Support Vector Regression

### Concept

SVR finds a hyperplane that fits data within a margin of tolerance.

**Mathematical Model**:
```
f(x) = w·φ(x) + b

Where:
- w: Weight vector
- φ(x): Feature transformation (kernel)
- b: Bias
- Goal: Minimize ||w|| subject to error ≤ ε
```

### Epsilon Tube

```
Price
  ^
  |      _____ ε-upper
  |    /       \
  |   |_________|  ← Support Vectors (boundary points)
  |  / Margin  \
  | /___________\
  |    / ε-lower
  └────────────────> Feature

Points within tube are not penalized
Points outside tube are penalized
```

### Advantages
- ✓ Effective in high dimensions
- ✓ Memory efficient (uses only support vectors)
- ✓ Flexible with different kernels
- ✓ Good for complex non-linear relationships
- ✓ Robust mathematical foundation

### Disadvantages
- ✗ Requires feature scaling (very important!)
- ✗ Hyperparameter tuning critical
- ✗ Slower than tree-based methods
- ✗ Less interpretable
- ✗ Sensitive to feature selection

### Kernels

```python
# Linear kernel: Linear relationship
SVR(kernel='linear')

# RBF (Radial Basis Function): Non-linear, most common
SVR(kernel='rbf', gamma='scale')

# Polynomial kernel: Polynomial relationships
SVR(kernel='poly', degree=3)

# Sigmoid kernel: Neural network-like
SVR(kernel='sigmoid')
```

### Key Parameters

```python
model = SVR(
    kernel='rbf',      # Kernel type
    C=100,             # Regularization parameter
    epsilon=0.1,       # Margin tolerance
    gamma='scale'      # Kernel coefficient
)

# C: Higher C → fit training data more closely (overfitting risk)
# epsilon: Size of error margin (higher = more tolerance)
# gamma: How far influence of single sample reaches
```

---

## Model Comparison

### Performance Characteristics

```
Model                 Speed   Accuracy   Interpretable   Complexity
────────────────────────────────────────────────────────────────────
Linear Regression     ★★★★★  ★★☆☆☆     ★★★★★          ★☆☆☆☆
Decision Tree         ★★★★☆  ★★★☆☆     ★★★★☆          ★★☆☆☆
Random Forest         ★★★☆☆  ★★★★★     ★★☆☆☆          ★★★☆☆
Gradient Boosting     ★★☆☆☆  ★★★★★     ★★☆☆☆          ★★★★☆
SVR                   ★★☆☆☆  ★★★★☆     ★☆☆☆☆          ★★★★★
```

### When to Use Each Model

**Linear Regression**:
- First baseline model
- When interpretability is critical
- Small datasets
- Linear relationships expected

**Decision Tree**:
- Quick exploration
- Need feature importance
- Mixed data types
- Need tree visualization

**Random Forest**:
- Good default algorithm
- Balanced accuracy and speed
- Feature interactions
- Don't care much about interpretability

**Gradient Boosting**:
- Need best possible performance
- Have time for hyperparameter tuning
- Complex non-linear patterns
- Kaggle competitions

**Support Vector Regression**:
- High-dimensional data
- Non-linear patterns
- Small to medium datasets
- Feature space is important

---

## Evaluation Metrics

### Mean Absolute Error (MAE)

**Formula**:
```
MAE = (1/n) × Σ|y_actual - y_pred|

Example:
Actual: [300k, 400k, 500k]
Pred:   [310k, 395k, 505k]
Errors: [10k, 5k, 5k]
MAE = (10k + 5k + 5k) / 3 = 6.67k
```

**Interpretation**:
- Average prediction error in dollars
- Same units as target variable
- Easy to understand and explain
- Not influenced by outliers as much as RMSE

### Root Mean Squared Error (RMSE)

**Formula**:
```
RMSE = √[(1/n) × Σ(y_actual - y_pred)²]

Example:
Errors: [10k, 5k, 5k]
Squared: [100M, 25M, 25M]
Mean: 50M
RMSE = √50M = 7.07k
```

**Interpretation**:
- Penalizes larger errors more
- Same units as target variable
- More sensitive to outliers than MAE
- Preferred in many ML contexts

### R² Score (Coefficient of Determination)

**Formula**:
```
R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(y_actual - y_pred)² (residual sum of squares)
- SS_tot = Σ(y_actual - y_mean)² (total sum of squares)
```

**Interpretation**:
- Proportion of variance explained by model
- Range: -∞ to 1 (1 is perfect)
- Example: R² = 0.85 means model explains 85% of price variation
- Higher is better

**Scale**:
- 1.0: Perfect prediction
- 0.8-0.9: Excellent
- 0.6-0.8: Good
- 0.4-0.6: Fair
- 0.0-0.4: Poor
- < 0: Worse than predicting mean

---

## Practical Guide: Choosing and Tuning Models

### Model Selection Process

```
1. START: Split data (80/20)
   ↓
2. Train Linear Regression (baseline)
   ↓
3. Calculate test metrics
   ↓
4. R² > 0.8? → Done! Use Linear Regression
   ↓ NO
5. Train Random Forest (quick improvement)
   ↓
6. R² > 0.85? → Use Random Forest
   ↓ NO
7. Train Gradient Boosting (complex tuning)
   ↓
8. Compare all models, select best
   ↓
9. Fine-tune hyperparameters of best model
   ↓
10. Final evaluation on test set
```

### Hyperparameter Tuning

**Grid Search Example**:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.5]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(),
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(
    model, X_train, y_train, 
    cv=5, 
    scoring='r2'
)

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

---

## Common Issues and Solutions

### Issue 1: Model Underfitting
**Symptoms**: Low training and test accuracy

**Solutions**:
- Use more complex model (tree → forest → boosting)
- Increase model capacity (max_depth, n_estimators)
- Get more training data
- Better feature engineering

### Issue 2: Model Overfitting
**Symptoms**: High training accuracy, low test accuracy

**Solutions**:
- Use simpler model (boosting → forest → tree)
- Add regularization (lower C for SVR, max_depth for trees)
- Get more training data
- Use cross-validation
- Remove noisy features

### Issue 3: Need Feature Scaling
**Symptoms**: SVR or distance-based model performs poorly

**Solutions**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y_train)
```

### Issue 4: Feature Importance
**Symptoms**: Don't know which features matter

**Solutions**:
```python
# For tree-based models
importances = model.feature_importances_
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")

# For Linear Regression
coefficients = model.coef_
for feature, coef in zip(feature_names, coefficients):
    print(f"{feature}: {coef:.4f}")
```

---

## Summary

| Model | Best For | Complexity | Tuning |
|-------|----------|-----------|--------|
| Linear Regression | Baseline, interpretability | Low | Minimal |
| Decision Tree | Quick exploration | Low | Depth |
| Random Forest | Good default | Medium | Trees, depth |
| Gradient Boosting | Best performance | High | Learning rate, depth |
| SVR | Complex non-linear | High | C, gamma, kernel |

**Remember**: Start simple (Linear), move to complex (Gradient Boosting) only if needed.

---

**Happy Modeling! 🚀**
