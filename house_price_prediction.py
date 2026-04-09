"""
================================================================================
                    HOUSE PRICE PREDICTION USING ML
                      Regression Model Implementation
================================================================================

This script implements multiple machine learning regression algorithms to predict
residential property prices based on various features and characteristics.

Dataset: Housing data with property features (square footage, bedrooms, bathrooms, etc.)
Target: Continuous price prediction (numeric value)
Models: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, SVR

Project Structure:
1. Data Collection & Loading - Import housing dataset
2. Exploratory Data Analysis - Understand data characteristics
3. Data Preprocessing - Handle missing values, scaling, encoding
4. Feature Engineering - Create new predictive features
5. Model Training - Train multiple regression algorithms
6. Model Evaluation - Compare performance metrics
7. Prediction System - Make predictions on new properties

Author: Machine Learning Project
License: MIT
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CLASS: HousePricePredictor
# ============================================================================

class HousePricePredictor:
    """
    A comprehensive class for house price prediction using multiple ML models.
    
    This class encapsulates the entire regression pipeline:
    - Data loading and exploration
    - Feature preprocessing and engineering
    - Multiple model training
    - Model evaluation and comparison
    - Price prediction on new properties
    
    Attributes:
        data (pd.DataFrame): Raw dataset
        X (np.ndarray): Features for modeling
        y (np.ndarray): Target prices
        X_train, X_test: Training and test features
        y_train, y_test: Training and test prices
        models (dict): Trained regression models
        scaler (StandardScaler): Feature scaler
        encoders (dict): Category encoders
    """
    
    def __init__(self):
        """Initialize the predictor with empty model storage."""
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.model_results = {}
        self.feature_names = None
        
    def load_data(self, filepath):
        """
        Load the housing dataset.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            print("=" * 70)
            print("STEP 1: DATA COLLECTION & LOADING")
            print("=" * 70)
            print(f"\n📂 Loading housing dataset from: {filepath}")
            
            self.data = pd.read_csv(filepath)
            
            print(f"✓ Dataset loaded successfully!")
            print(f"  - Total Records: {self.data.shape[0]}")
            print(f"  - Total Features: {self.data.shape[1]}")
            print(f"  - Dataset Shape: {self.data.shape}\n")
            
            return self.data
            
        except FileNotFoundError:
            print(f"❌ Error: File '{filepath}' not found!")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def explore_data(self):
        """
        Perform comprehensive exploratory data analysis.
        
        Displays:
        - Dataset preview
        - Statistical summary
        - Data types and missing values
        - Price distribution
        - Feature correlations
        """
        if self.data is None:
            print("❌ Error: Data not loaded. Call load_data() first.")
            return
        
        print("\n" + "=" * 70)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 70)
        
        # Dataset preview
        print("\n📊 DATASET PREVIEW (First 5 rows):")
        print("-" * 70)
        print(self.data.head())
        
        # Data info
        print("\n\n📋 DATA INFORMATION:")
        print("-" * 70)
        print(f"Total Rows: {self.data.shape[0]}")
        print(f"Total Columns: {self.data.shape[1]}")
        print(f"\nData Types:")
        print(self.data.dtypes)
        
        # Missing values
        print("\n\n⚠️ MISSING VALUES:")
        print("-" * 70)
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
            print(f"\nTotal missing values: {missing.sum()}")
        else:
            print("✓ No missing values detected!")
        
        # Statistical summary
        print("\n\n📈 STATISTICAL SUMMARY:")
        print("-" * 70)
        print(self.data.describe())
        
        # Identify numeric and categorical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Target variable analysis (assuming last numeric column is price)
        if numeric_cols:
            target_col = numeric_cols[-1]
            print(f"\n\n💰 TARGET VARIABLE (PRICE) ANALYSIS:")
            print("-" * 70)
            print(f"Column: {target_col}")
            print(f"  - Min Price: ${self.data[target_col].min():,.2f}")
            print(f"  - Max Price: ${self.data[target_col].max():,.2f}")
            print(f"  - Mean Price: ${self.data[target_col].mean():,.2f}")
            print(f"  - Median Price: ${self.data[target_col].median():,.2f}")
            print(f"  - Std Dev: ${self.data[target_col].std():,.2f}")
            print(f"  - Skewness: {self.data[target_col].skew():.4f}")
            
            # Price distribution
            print(f"\n  Price Distribution:")
            q1, q3 = self.data[target_col].quantile([0.25, 0.75])
            print(f"  - Q1 (25%): ${q1:,.2f}")
            print(f"  - Q2 (50%): ${self.data[target_col].median():,.2f}")
            print(f"  - Q3 (75%): ${q3:,.2f}")
        
        # Feature analysis
        print(f"\n\n🔍 FEATURE ANALYSIS:")
        print("-" * 70)
        print(f"Numeric Features ({len(numeric_cols)}):")
        for col in numeric_cols:
            print(f"  - {col}: {self.data[col].dtype}")
        
        if categorical_cols:
            print(f"\nCategorical Features ({len(categorical_cols)}):")
            for col in categorical_cols:
                unique_count = self.data[col].nunique()
                print(f"  - {col}: {unique_count} unique values")
        
        # Correlation analysis (numeric only)
        if len(numeric_cols) > 1:
            print(f"\n\n📊 CORRELATION WITH TARGET:")
            print("-" * 70)
            target = numeric_cols[-1]
            correlations = self.data[numeric_cols].corr()[target].drop(target).sort_values(ascending=False)
            print(correlations)
        
        print("\n✓ EDA completed!\n")
    
    def preprocess_data(self, target_col=None, test_size=0.2, random_state=42):
        """
        Preprocess data: handle missing values, scaling, and encoding.
        
        Args:
            target_col (str): Name of target column (default: last numeric)
            test_size (float): Proportion for test set
            random_state (int): Random seed for reproducibility
        """
        if self.data is None:
            print("❌ Error: Data not loaded. Call load_data() first.")
            return
        
        print("=" * 70)
        print("STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING")
        print("=" * 70)
        
        # Make a copy
        df = self.data.copy()
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Set target column
        if target_col is None:
            target_col = numeric_cols[-1]
        
        print(f"\n🎯 Target Column: {target_col}")
        print(f"   Numeric Features: {len(numeric_cols)-1}")
        print(f"   Categorical Features: {len(categorical_cols)}")
        
        # Step 1: Handle missing values
        print(f"\n📌 Handling Missing Values...")
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            print(f"   Found {missing_count} missing values")
            # Fill numeric with median
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            # Fill categorical with mode
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"   ✓ Missing values handled!")
        else:
            print(f"   ✓ No missing values found!")
        
        # Step 2: Remove outliers (optional - comment out if needed)
        print(f"\n📌 Outlier Detection...")
        initial_rows = len(df)
        
        # For each numeric column, remove extreme outliers (beyond 3 std dev)
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]
        
        removed = initial_rows - len(df)
        if removed > 0:
            print(f"   ✓ Removed {removed} outlier rows")
        else:
            print(f"   ✓ No extreme outliers detected")
        
        # Step 3: Encode categorical variables
        print(f"\n📌 Encoding Categorical Variables...")
        if categorical_cols:
            for col in categorical_cols:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])
                self.encoders[col] = encoder
            print(f"   ✓ Encoded {len(categorical_cols)} categorical features")
        
        # Step 4: Separate features and target
        print(f"\n📌 Separating Features and Target...")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        self.feature_names = df.drop(columns=[target_col]).columns.tolist()
        
        print(f"   - Features shape: {X.shape}")
        print(f"   - Target shape: {y.shape}")
        
        # Step 5: Feature scaling
        print(f"\n📌 Scaling Features (StandardScaler)...")
        X_scaled = self.scaler.fit_transform(X)
        print(f"   ✓ Features scaled (mean=0, std=1)")
        
        # Step 6: Train-test split
        print(f"\n📌 Train-Test Split...")
        print(f"   - Training set: {100 - (test_size*100):.0f}% ({int(len(X_scaled) * (1-test_size))} samples)")
        print(f"   - Test set: {test_size*100:.0f}% ({int(len(X_scaled) * test_size)} samples)")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"   - X_train shape: {self.X_train.shape}")
        print(f"   - X_test shape: {self.X_test.shape}")
        print(f"\n✓ Data preprocessing completed!\n")
    
    def train_models(self):
        """Train multiple regression models."""
        if self.X_train is None:
            print("❌ Error: Data not preprocessed. Call preprocess_data() first.")
            return
        
        print("=" * 70)
        print("STEP 4: MODEL TRAINING")
        print("=" * 70)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'Support Vector Regressor': SVR(kernel='rbf', C=100, gamma='scale')
        }
        
        print(f"\nTraining {len(models)} regression models...\n")
        
        for name, model in models.items():
            print(f"🤖 Training {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                self.models[name] = model
                print(f"   ✓ {name} trained successfully!\n")
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}\n")
        
        print(f"✓ Trained {len(self.models)} models!\n")
    
    def evaluate_models(self):
        """Evaluate all trained models and compare performance."""
        if not self.models:
            print("❌ Error: No models trained. Call train_models() first.")
            return
        
        print("=" * 70)
        print("STEP 5: MODEL EVALUATION & COMPARISON")
        print("=" * 70)
        
        print("\n📊 REGRESSION METRICS EXPLANATION:")
        print("-" * 70)
        print("MAE (Mean Absolute Error):")
        print("  - Average absolute difference between predicted and actual")
        print("  - Units: Same as target (dollars)")
        print("  - Lower is better\n")
        
        print("RMSE (Root Mean Squared Error):")
        print("  - Square root of average squared errors")
        print("  - Penalizes larger errors more")
        print("  - Units: Same as target (dollars)")
        print("  - Lower is better\n")
        
        print("R² Score (Coefficient of Determination):")
        print("  - Proportion of variance explained by model")
        print("  - Range: -∞ to 1 (1 is perfect)")
        print("  - Higher is better")
        print("  - Example: 0.85 = model explains 85% of price variation\n")
        
        results = {}
        
        print("\n" + "=" * 70)
        print("INDIVIDUAL MODEL PERFORMANCE")
        print("=" * 70)
        
        for name, model in self.models.items():
            print(f"\n📈 {name}:")
            print("-" * 70)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Store results
            results[name] = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
            # Display results
            print(f"  TRAINING METRICS:")
            print(f"    MAE:  ${train_mae:,.2f}")
            print(f"    RMSE: ${train_rmse:,.2f}")
            print(f"    R²:   {train_r2:.4f}")
            
            print(f"\n  TEST METRICS (Generalization):")
            print(f"    MAE:  ${test_mae:,.2f}")
            print(f"    RMSE: ${test_rmse:,.2f}")
            print(f"    R²:   {test_r2:.4f}")
            
            # Overfitting check
            mae_diff = train_mae - test_mae
            r2_diff = train_r2 - test_r2
            print(f"\n  OVERFITTING CHECK:")
            print(f"    MAE Difference: ${mae_diff:,.2f}")
            print(f"    R² Difference: {r2_diff:.4f}")
            
            if r2_diff < 0.05:
                status = "✓ Good generalization"
            elif r2_diff < 0.10:
                status = "⚠ Slight overfitting"
            else:
                status = "⚠ Significant overfitting"
            print(f"    Status: {status}")
        
        self.model_results = results
        
        # Model comparison
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        print("\n📊 TEST SET PERFORMANCE RANKING:")
        print("-" * 70)
        
        # Sort by test R²
        sorted_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        print(f"\n{'Rank':<6}{'Model':<25}{'Test MAE':<18}{'Test RMSE':<18}{'Test R²':<10}")
        print("-" * 70)
        
        for rank, (name, metrics) in enumerate(sorted_models, 1):
            print(f"{rank:<6}{name:<25}${metrics['test_mae']:<17,.2f}${metrics['test_rmse']:<17,.2f}{metrics['test_r2']:<10.4f}")
        
        # Best model
        best_model_name = sorted_models[0][0]
        best_metrics = sorted_models[0][1]
        
        print(f"\n✓ Best Model: {best_model_name}")
        print(f"  - Test R² Score: {best_metrics['test_r2']:.4f}")
        print(f"  - Test MAE: ${best_metrics['test_mae']:,.2f}")
        print(f"  - Test RMSE: ${best_metrics['test_rmse']:,.2f}")
        
        print("\n✓ Model evaluation completed!\n")
    
    def predict_price(self, property_features):
        """
        Predict house price using the best model.
        
        Args:
            property_features (dict): Property features as dictionary
                
        Returns:
            float: Predicted price
        """
        if not self.models:
            print("❌ Error: No models trained.")
            return None
        
        # Get best model
        sorted_results = sorted(self.model_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        best_model_name = sorted_results[0][0]
        best_model = self.models[best_model_name]
        
        # Convert to array and scale
        features_array = np.array([property_features[col] if col in property_features else 0 
                                   for col in self.feature_names]).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Predict
        prediction = best_model.predict(features_scaled)[0]
        
        return prediction
    
    def run_complete_pipeline(self, filepath):
        """Run the complete machine learning pipeline."""
        print("\n")
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "    HOUSE PRICE PREDICTION - MACHINE LEARNING".center(68) + "║")
        print("║" + "    Multiple Regression Models".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "=" * 68 + "╝")
        
        # Run all steps
        self.load_data(filepath)
        self.explore_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        
        print("\n" + "=" * 70)
        print("✓ COMPLETE PIPELINE EXECUTION FINISHED")
        print("=" * 70)
        
        # Display best model summary
        if self.model_results:
            sorted_results = sorted(self.model_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
            best_name, best_metrics = sorted_results[0]
            print(f"\n✨ Best Model: {best_name}")
            print(f"   R² Score: {best_metrics['test_r2']:.4f}")
            print(f"   MAE: ${best_metrics['test_mae']:,.2f}")
            print(f"\n💾 Model is ready for predictions!\n")


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def create_sample_data(filename='housing_sample.csv', n_samples=200):
    """
    Create sample housing data for testing.
    
    Args:
        filename (str): Output CSV filename
        n_samples (int): Number of samples to generate
    """
    np.random.seed(42)
    
    data = {
        'square_footage': np.random.randint(1000, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.uniform(1, 4, n_samples),
        'age': np.random.randint(0, 100, n_samples),
        'garage_spaces': np.random.randint(0, 4, n_samples),
        'lot_size': np.random.randint(2000, 20000, n_samples),
        'location': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples),
    }
    
    # Generate price based on features
    df = pd.DataFrame(data)
    price = (
        150 * df['square_footage'] +
        50000 * df['bedrooms'] +
        30000 * df['bathrooms'] -
        2000 * df['age'] +
        20000 * df['garage_spaces'] +
        2 * df['lot_size'] +
        100000 * (df['location'] == 'Downtown').astype(int) +
        np.random.normal(0, 50000, n_samples)
    )
    
    df['price'] = price.astype(int)
    df.to_csv(filename, index=False)
    print(f"✓ Sample data created: {filename}")
    
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the complete house price prediction pipeline."""
    
    # Create predictor instance
    predictor = HousePricePredictor()
    
    # Try to load existing data, or create sample
    try:
        predictor.run_complete_pipeline('housing.csv')
    except FileNotFoundError:
        print("\n📝 Creating sample housing dataset for demonstration...")
        create_sample_data('housing_sample.csv')
        print("\n🔄 Running pipeline with sample data...\n")
        predictor.run_complete_pipeline('housing_sample.csv')


if __name__ == "__main__":
    main()
