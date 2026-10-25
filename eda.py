import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading dataset...")
    # Load the training data
    train_df = pd.read_csv('data/train.csv')

    # Display the first 5 rows
    print("\n--- First 5 Rows ---")
    print(train_df.head())

    # Check shape and missing values
    print("\n--- Dataset Shape ---")
    print(train_df.shape)

    print("\n--- Columns with Missing Values ---")
    missing = train_df.isnull().sum()
    print(missing[missing > 0])

    # Optional: Quick visualization of the target variable (SalePrice)
    plt.figure(figsize=(8, 5))
    sns.histplot(train_df['SalePrice'], kde=True, color='blue')
    plt.title('Distribution of Sale Price')
    plt.xlabel('SalePrice')
    plt.ylabel('Count')
    
    # Note: In a .py script, plt.show() opens a popup window with the plot
    print("\nDisplaying SalePrice distribution plot...")
    plt.show()

if __name__ == "__main__":
    main()