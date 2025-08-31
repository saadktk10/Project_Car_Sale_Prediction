
import pickle
import numpy as np

print("🚗 Car Price Prediction Tool")
print("============================")
print("Please enter the car details:\n")

# Get user input
year = int(input("Manufacturing Year (e.g., 2020): "))
engine = int(input("Engine Size (cc, e.g., 1500): "))
mileage = int(input("Mileage (km, e.g., 50000): "))
Car_type = input("Car Type (e.g., Sedan, SUV): ")
Manufacturer = input("Manufacturer (e.g., Toyota, Honda): ")
# Load model from correct path
try:
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model file not found! Make sure you're in the main project directory.")
    exit()

# Make prediction
features = [year, engine, mileage]
predicted_price = model.predict([features])[0]

print(f"\n🎯 Prediction Results:")
print(f"Year: {year}")
print(f"Engine: {engine}cc")
print(f"Mileage: {mileage}km")
print(f"Car Type: {Car_type}")
print(f"Manufacturer: {Manufacturer}")
print(f"Predicted Price: Rs{predicted_price:,.2f}")
print(f"Approx USD: ${predicted_price/75:,.2f}")
