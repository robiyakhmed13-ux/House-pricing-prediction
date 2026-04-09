# Quick Start Guide - House Price Prediction

## 🏠 Get Started in 5 Minutes

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Step 2: Prepare Your Data

Your housing dataset should be a CSV file with columns like:
```
square_footage,bedrooms,bathrooms,age,garage_spaces,lot_size,location,price
2500,4,2.5,10,2,7500,Downtown,450000
1800,3,2,15,1,5000,Suburbs,350000
...
```

**Column Format**:
- Numeric columns: Directly usable
- Categorical columns: Automatically encoded
- Target column: Should be named 'price' (or last column)

### Step 3: Run the Script

```bash
python house_price_prediction.py
```

### Step 4: View Results

The script will output:
1. Dataset statistics and visualization
2. Missing value detection
3. Model training progress
4. Performance comparison
5. Best model identification

---

## 📊 Expected Output

```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        HOUSE PRICE PREDICTION - MACHINE LEARNING                 ║
║            Multiple Regression Models                             ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

======================================================================
STEP 1: DATA COLLECTION & LOADING
======================================================================

📂 Loading housing dataset from: housing.csv
✓ Dataset loaded successfully!
  - Total Records: 200
  - Total Features: 8
  - Dataset Shape: (200, 8)

...

======================================================================
STEP 4: MODEL TRAINING
======================================================================

Training 5 regression models...

🤖 Training Linear Regression...
   ✓ Linear Regression trained successfully!

🤖 Training Decision Tree...
   ✓ Decision Tree trained successfully!

...

======================================================================
STEP 5: MODEL EVALUATION & COMPARISON
======================================================================

📊 TEST SET PERFORMANCE RANKING:
──────────────────────────────────────────────────────────────────────

Rank  Model                    Test MAE           Test RMSE           Test R²
──────────────────────────────────────────────────────────────────────
1     Random Forest            $25,000.00         $32,000.00          0.9200
2     Gradient Boosting        $28,000.00         $35,000.00          0.9100
3     Decision Tree            $35,000.00         $42,000.00          0.8800
4     SVR                      $40,000.00         $48,000.00          0.8500
5     Linear Regression        $50,000.00         $62,000.00          0.8000

✓ Best Model: Random Forest
  - Test R² Score: 0.9200
  - Test MAE: $25,000.00
  - Test RMSE: $32,000.00

💾 Model is ready for predictions!
```

---

## 💻 Using the Predictor Class

### Method 1: Full Pipeline

```python
from house_price_prediction import HousePricePredictor

# Create predictor
predictor = HousePricePredictor()

# Run complete pipeline
predictor.run_complete_pipeline('housing.csv')
```

### Method 2: Step by Step

```python
from house_price_prediction import HousePricePredictor

predictor = HousePricePredictor()

# 1. Load data
predictor.load_data('housing.csv')

# 2. Explore
predictor.explore_data()

# 3. Preprocess
predictor.preprocess_data()

# 4. Train models
predictor.train_models()

# 5. Evaluate
predictor.evaluate_models()

# 6. Make predictions
price = predictor.predict_price({
    'square_footage': 2500,
    'bedrooms': 4,
    'bathrooms': 2.5,
    'age': 10,
    'garage_spaces': 2,
    'lot_size': 7500,
    'location_encoded': 1  # Encoded as number
})

print(f"Predicted Price: ${price:,.2f}")
```

### Method 3: Using Sample Data

```python
from house_price_prediction import create_sample_data, HousePricePredictor

# Create sample data
create_sample_data('housing_sample.csv', n_samples=500)

# Run predictor
predictor = HousePricePredictor()
predictor.run_complete_pipeline('housing_sample.csv')
```

---

## 📊 Dataset Format

### Minimum Dataset
Your CSV must have at least:
- Features: Property characteristics (numeric or categorical)
- Target: House price (numeric)

### Example Formats

**Basic Format**:
```csv
square_footage,bedrooms,bathrooms,price
2500,4,2.5,450000
1800,3,2,350000
```

**Extended Format**:
```csv
square_footage,bedrooms,bathrooms,age,garage_spaces,lot_size,location,price
2500,4,2.5,10,2,7500,Downtown,450000
1800,3,2,15,1,5000,Suburbs,350000
1200,2,1,30,1,3000,Rural,250000
```

**Real Data Format** (e.g., Boston Housing):
```csv
CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV
0.02731,0,7.07,0,0.469,6.421,78.9,4.9671,2,242,17.8,396.9,9.14,24.0
0.02729,0,7.07,0,0.469,7.185,61.1,4.9671,2,242,17.8,392.83,4.61,21.6
```

---

## 🔧 Understanding the Output

### MAE (Mean Absolute Error)
- **What it is**: Average prediction error
- **Interpretation**: "On average, predictions are $25,000 off"
- **Lower is better**: $10,000 is better than $50,000

### RMSE (Root Mean Squared Error)
- **What it is**: Penalizes larger errors more
- **Interpretation**: "Typical error is $32,000"
- **Lower is better**: Accounts for big prediction mistakes

### R² Score
- **What it is**: % of price variation model explains
- **Interpretation**: "0.92 = Model explains 92% of price variation"
- **Range**: 0 to 1 (higher is better)
- **Typical**: 0.8-0.95 is excellent for house prices

---

## 🎯 Common Use Cases

### Use Case 1: Quick Evaluation
```bash
python house_price_prediction.py
```
See which model performs best on your data.

### Use Case 2: Make Single Prediction
```python
from house_price_prediction import HousePricePredictor

predictor = HousePricePredictor()
predictor.run_complete_pipeline('housing.csv')

# Predict new property
new_house = {
    'square_footage': 3000,
    'bedrooms': 5,
    'bathrooms': 3,
    'age': 5,
    'garage_spaces': 3,
    'lot_size': 10000,
    'location_encoded': 1
}

predicted_price = predictor.predict_price(new_house)
print(f"Estimated Price: ${predicted_price:,.0f}")
```

### Use Case 3: Batch Predictions
```python
import pandas as pd
from house_price_prediction import HousePricePredictor

predictor = HousePricePredictor()
predictor.run_complete_pipeline('training_data.csv')

# Load new properties
new_properties = pd.read_csv('new_properties.csv')

# Predict for each
predictions = []
for idx, row in new_properties.iterrows():
    property_dict = row.to_dict()
    price = predictor.predict_price(property_dict)
    predictions.append(price)

# Save results
new_properties['predicted_price'] = predictions
new_properties.to_csv('predictions.csv', index=False)
```

---

## 🐛 Troubleshooting

### Error: "FileNotFoundError: housing.csv"
**Solution**: Place your CSV file in the project root directory, or specify full path:
```python
predictor.run_complete_pipeline('/path/to/your/housing.csv')
```

### Error: "No module named 'sklearn'"
**Solution**: Install scikit-learn
```bash
pip install scikit-learn
```

### Low Model Accuracy (R² < 0.7)
**Possible causes**:
1. **Poor data quality**
   - Fix: Remove outliers, handle missing values properly
   
2. **Missing important features**
   - Fix: Add features that better predict price (location, condition, etc.)
   
3. **Categorical variable issues**
   - Fix: Ensure categorical columns are properly encoded

4. **Too few samples**
   - Fix: Collect more data (typically need 100+ samples)

**Solutions**:
```python
# Check data distribution
predictor.explore_data()

# Try feature engineering
df['price_per_sqft'] = df['price'] / df['square_footage']
df['room_count'] = df['bedrooms'] + df['bathrooms']

# Check for outliers
prices = df['price']
q1, q3 = prices.quantile([0.25, 0.75])
iqr = q3 - q1
outliers = (prices < q1 - 3*iqr) | (prices > q3 + 3*iqr)
print(f"Found {outliers.sum()} outliers")
```

### Predictions Seem Unrealistic
**Check**:
1. Are features scaled correctly?
2. Are categorical variables encoded consistently?
3. Is the model trained on similar data?

**Example**:
```python
# If predicting $500,000 house but model predicts $50,000:
# Check if all inputs are in same units/scale as training data

training_sqft = predictor.X_train[:, 0]  # Square footage feature
print(f"Training sqft range: {training_sqft.min()}-{training_sqft.max()}")

# Make sure new property sqft is in same range
```

---

## 📈 Performance Tips

### Improve Model Accuracy
1. **Better Features**
   ```python
   # Create interaction features
   df['rooms_per_bath'] = df['bedrooms'] / df['bathrooms']
   df['price_per_sqft'] = df['price'] / df['square_footage']
   ```

2. **Remove Outliers**
   ```python
   # Remove extreme prices
   df = df[(df['price'] > df['price'].quantile(0.01)) & 
           (df['price'] < df['price'].quantile(0.99))]
   ```

3. **More Data**
   ```python
   # Models improve with more training samples
   # Aim for 500+ samples for good performance
   ```

4. **Hyperparameter Tuning**
   ```python
   # Adjust model parameters
   model = GradientBoostingRegressor(
       n_estimators=200,      # More trees
       learning_rate=0.05,    # Slower learning
       max_depth=7            # Deeper trees
   )
   ```

### Speed Up Training
```python
# Use fewer features
# Reduce model complexity
# Sample data during development

sample_data = df.sample(frac=0.2)  # 20% sample
predictor.run_complete_pipeline('sample.csv')
```

---

## 📝 Sample Predictions

### Property 1: Modest House
```
Input:
  - Square Footage: 1,500
  - Bedrooms: 3
  - Bathrooms: 1.5
  - Age: 25 years
  - Garage: 1 space
  - Location: Suburbs

Expected Price Range: $200,000 - $300,000
```

### Property 2: Premium House
```
Input:
  - Square Footage: 4,000
  - Bedrooms: 5
  - Bathrooms: 4
  - Age: 2 years
  - Garage: 3 spaces
  - Location: Downtown

Expected Price Range: $600,000 - $800,000
```

### Property 3: Average House
```
Input:
  - Square Footage: 2,500
  - Bedrooms: 4
  - Bathrooms: 2.5
  - Age: 10 years
  - Garage: 2 spaces
  - Location: Suburbs

Expected Price Range: $400,000 - $500,000
```

---

## 🎓 Learning Resources

- **Scikit-learn Regression Guide**: https://scikit-learn.org/stable/modules/regression.html
- **Pandas Tutorial**: https://pandas.pydata.org/
- **NumPy Basics**: https://numpy.org/doc/stable/user/basics.html
- **Feature Engineering**: https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/

---

## 📊 Next Steps

1. **Evaluate Models**: See which performs best
2. **Fine-tune Best Model**: Adjust hyperparameters
3. **Cross-validate**: Use K-fold validation
4. **Save Model**: Pickle for production
5. **Deploy**: Create API or web interface

---

**Happy Predicting! 🏠💰**
