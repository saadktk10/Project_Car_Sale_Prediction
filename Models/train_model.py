import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def load_data():
    """Load the dataset"""
    df = pd.read_csv('../data/pakwheels_used_car_data.csv')
    return df

def preprocess_data(df):
    """Preprocess the data for training"""
    # Make a copy of the dataframe
    data = df.copy()
    
    # Handle missing values
    data['mileage'].fillna(data['mileage'].median(), inplace=True)
    data['engine'].fillna(data['engine'].median(), inplace=True)
    data['year'].fillna(data['year'].median(), inplace=True)
    
    # Fill categorical missing values with 'Unknown'
    categorical_cols = ['assembly', 'body', 'color', 'registered', 'transmission', 'fuel']
    for col in categorical_cols:
        data[col].fillna('Unknown', inplace=True)
    
    # Remove rows with missing price (our target variable)
    data = data.dropna(subset=['price'])
    
    # Remove outliers in price and mileage
    data = data[(data['price'] > 100000) & (data['price'] < 50000000)]
    data = data[data['mileage'] < 500000]
    
    # Feature engineering
    data['car_age'] = 2023 - data['year']  # Assuming current year is 2023
    data['price_per_cc'] = data['price'] / data['engine']
    data['mileage_per_year'] = data['mileage'] / (data['car_age'] + 1)  # +1 to avoid division by zero
    
    return data

def encode_categorical_features(df):
    """Encode categorical features and return encoders"""
    categorical_features = ['make', 'model', 'city', 'assembly', 'body', 'transmission', 'fuel', 'color', 'registered']
    label_encoders = {}
    
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
        label_encoders[feature] = le
    
    return df, label_encoders

def train_model():
    """Train the machine learning model"""
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Encoding categorical features...")
    df, label_encoders = encode_categorical_features(df)
    
    # Define features and target
    features = ['make', 'model', 'year', 'engine', 'transmission', 'fuel', 
                'mileage', 'city', 'assembly', 'body', 'color', 'registered',
                'car_age', 'mileage_per_year']
    
    X = df[features]
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation:")
    print(f"MAE: {mae:,.2f}")
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    
    # Save the model and encoders
    print("Saving model and encoders...")
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    print("Training complete! Model saved to models/ directory.")

if __name__ == "__main__":
    train_model()