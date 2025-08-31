import pandas as pd
import numpy as np

def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset"""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df['mileage'].fillna(df['mileage'].median(), inplace=True)
    df['engine'].fillna(df['engine'].median(), inplace=True)
    df['year'].fillna(df['year'].median(), inplace=True)
    
    # Fill categorical missing values with 'Unknown'
    categorical_cols = ['assembly', 'body', 'color', 'registered', 'transmission', 'fuel']
    for col in categorical_cols:
        df[col].fillna('Unknown', inplace=True)
    
    # Remove rows with missing price
    df = df.dropna(subset=['price'])
    
    return df

def remove_outliers(df, column, lower_percentile=0.01, upper_percentile=0.99):
    """Remove outliers from a column"""
    lower_bound = df[column].quantile(lower_percentile)
    upper_bound = df[column].quantile(upper_percentile)
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def get_unique_values(df, column):
    """Get unique values from a column"""
    return sorted(df[column].unique())

def get_average_prices_by_make(df):
    """Get average prices by car make"""
    return df.groupby('make')['price'].mean().sort_values(ascending=False)

def get_popular_models(df, make, top_n=5):
    """Get most popular models for a given make"""
    return df[df['make'] == make]['model'].value_counts().head(top_n).index.tolist()