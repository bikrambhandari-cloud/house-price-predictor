import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from preprocess import preprocess_data

def main():
    # 1. Get preprocessed data from our preprocess script
    X, y = preprocess_data()
    
    # 2. Split into Train (80%) and Validation (20%) sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # 3. Train Random Forest Model
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    rf_preds = rf_model.predict(X_val)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
    print(f"Random Forest Validation RMSE: ${rf_rmse:,.2f}")
    
    # 4. Train XGBoost Model
    print("\nTraining XGBoost Regressor...")
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    xgb_preds = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))
    print(f"XGBoost Validation RMSE: ${xgb_rmse:,.2f}")

if __name__ == "__main__":
    main()