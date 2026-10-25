import pandas as pd
import numpy as np

def preprocess_data():
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    
    # 1. Drop the 'Id' column since it's just an identifier, not a feature
    train_df = train_df.drop('Id', axis=1)
    
    # 2. Separate features (X) and target (y)
    X = train_df.drop('SalePrice', axis=1)
    y = train_df['SalePrice']
    
    # 3. Handle Missing Values
    # For numerical columns, fill missing values with the median
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    # For categorical columns, fill missing values with the string 'None' or 'Missing'
    cat_cols = X.select_dtypes(include=['object']).columns
    X[cat_cols] = X[cat_cols].fillna('Missing')
    
    # 4. Encode Categorical Variables (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Processed feature matrix shape: {X.shape}")
    print("Preprocessing complete! Ready for model training.")
    
    return X, y

if __name__ == "__main__":
    preprocess_data()