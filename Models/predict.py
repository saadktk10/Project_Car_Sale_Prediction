import pandas as pd
import numpy as np
import pickle

def load_model():
    """Load the trained model and encoders"""
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    return model, scaler, label_encoders

def predict_price(input_data, model, scaler, label_encoders):
    """Predict car price based on input data"""
    # Create a copy of the input data
    processed_data = input_data.copy()
    
    # Encode categorical variables
    categorical_features = ['make', 'model', 'city', 'assembly', 'body', 'transmission', 'fuel', 'color', 'registered']
    
    for feature in categorical_features:
        if feature in processed_data:
            try:
                processed_data[feature] = label_encoders[feature].transform([processed_data[feature]])[0]
            except ValueError:
                # If value not in encoder, use the first available value
                processed_data[feature] = 0
    
    # Calculate derived features
    processed_data['car_age'] = 2023 - processed_data['year']
    processed_data['mileage_per_year'] = processed_data['mileage'] / max(1, processed_data['car_age'])
    
    # Create feature array in the right order
    features = ['make', 'model', 'year', 'engine', 'transmission', 'fuel', 
                'mileage', 'city', 'assembly', 'body', 'color', 'registered',
                'car_age', 'mileage_per_year']
    
    input_array = np.array([processed_data[feature] for feature in features]).reshape(1, -1)
    
    # Scale the input
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

if __name__ == "__main__":
    # Example usage
    model, scaler, label_encoders = load_model()
    
    # Example input data
    input_data = {
        'make': 'Toyota',
        'model': 'Corolla',
        'year': 2018,
        'engine': 1800,
        'transmission': 'Automatic',
        'fuel': 'Petrol',
        'mileage': 50000,
        'city': 'Lahore',
        'assembly': 'Local',
        'body': 'Sedan',
        'color': 'White',
        'registered': 'Lahore'
    }
    
    prediction = predict_price(input_data, model, scaler, label_encoders)
    print(f"Predicted price: ₨ {prediction:,.0f}")
    
   