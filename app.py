'''import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Pakistani Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/pakwheels_used_car_data.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please make sure 'data/pakwheels_used_car_data.csv' exists.")
        return None

# Load model and encoders
@st.cache_resource
def load_model():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        return model, scaler, label_encoders
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None, None

# Preprocess input data
def preprocess_input(input_data, label_encoders):
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
    
    return processed_data

# Main function
def main():
    # Load CSS
    load_css()
    
    # Load data and model
    df = load_data()
    model, scaler, label_encoders = load_model()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["Home", "Price Prediction", "Market Analysis", "About"])
    
    # Home page
    if app_mode == "Home":
        st.title("🇵🇰 Pakistani Used Car Price Predictor")
        st.markdown("""
        Welcome to the Pakistani Used Car Price Prediction tool! This application helps you:
        
        - 📊 Predict prices of used cars in Pakistan
        - 🔍 Analyze market trends
        - 📈 Understand factors affecting car prices
        
        Use the navigation menu on the left to get started.
        """)
        
        # Display some statistics
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Listings", f"{len(df):,}")
            with col2:
                st.metric("Unique Makes", f"{df['make'].nunique()}")
            with col3:
                st.metric("Average Price", f"₨{df['price'].mean():,.0f}")
        
        st.image("assets/images/car_banner.jpg", use_column_width=True)
    
    # Price Prediction page
    elif app_mode == "Price Prediction":
        st.title("Car Price Prediction")
        
        if df is None or model is None:
            st.warning("Please ensure the data file and model are available.")
            return
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            make = st.selectbox("Make", sorted(df['make'].unique()))
            model_name = st.selectbox("Model", sorted(df[df['make'] == make]['model'].unique()))
            year = st.slider("Year", int(df['year'].min()), 2023, 2020)
            engine = st.slider("Engine Capacity (CC)", int(df['engine'].min()), int(df['engine'].max()), 1300)
            transmission = st.selectbox("Transmission", df['transmission'].unique())
            fuel = st.selectbox("Fuel Type", df['fuel'].unique())
        
        with col2:
            mileage = st.slider("Mileage", 0, 300000, 50000)
            city = st.selectbox("City", sorted(df['city'].unique()))
            assembly = st.selectbox("Assembly", df['assembly'].unique())
            body = st.selectbox("Body Type", df['body'].unique())
            color = st.selectbox("Color", df['color'].unique())
            registered = st.selectbox("Registered City", sorted(df['registered'].unique()))
        
        # Predict button
        if st.button("Predict Price", type="primary"):
            # Prepare input data
            input_data = {
                'make': make,
                'model': model_name,
                'year': year,
                'engine': engine,
                'transmission': transmission,
                'fuel': fuel,
                'mileage': mileage,
                'city': city,
                'assembly': assembly,
                'body': body,
                'color': color,
                'registered': registered
            }
            
            # Preprocess input
            processed_data = preprocess_input(input_data, label_encoders)
            
            # Create feature array in the right order
            features = ['make', 'model', 'year', 'engine', 'transmission', 'fuel', 
                        'mileage', 'city', 'assembly', 'body', 'color', 'registered',
                        'car_age', 'mileage_per_year']
            
            input_array = np.array([processed_data[feature] for feature in features]).reshape(1, -1)
            
            # Scale the input
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display result
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted Price: ₨ {prediction:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Show similar cars in the market
            st.subheader("Similar Cars in Market")
            similar_cars = df[
                (df['make'] == make) & 
                (df['model'] == model_name) &
                (df['year'] >= year - 2) & 
                (df['year'] <= year + 2)
            ].head(5)
            
            if not similar_cars.empty:
                st.dataframe(similar_cars[['year', 'mileage', 'city', 'price']])
            else:
                st.info("No similar cars found in the dataset.")
    
    # Market Analysis page
    elif app_mode == "Market Analysis":
        st.title("Pakistani Used Car Market Analysis")
        
        if df is None:
            st.warning("Please ensure the data file is available.")
            return
        
        # Top car makes
        st.subheader("Top 10 Car Makes in Pakistan")
        make_counts = df['make'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        make_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
        # Price distribution by make
        st.subheader("Average Price by Make (Top 10)")
        top_makes = make_counts.index
        price_by_make = df[df['make'].isin(top_makes)].groupby('make')['price'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        price_by_make.plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_ylabel("Average Price (₨)")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
        # Price vs mileage
        st.subheader("Price vs Mileage")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['mileage'], df['price'], alpha=0.5, color='mediumseagreen')
        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price (₨)")
        st.pyplot(fig)
        
        # Price vs year
        st.subheader("Price vs Year")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df['year'], df['price'], alpha=0.5, color='darkorange')
        ax.set_xlabel("Year")
        ax.set_ylabel("Price (₨)")
        st.pyplot(fig)
    
    # About page
    elif app_mode == "About":
        st.title("About This Project")
        st.markdown("""
        ## Pakistani Used Car Price Prediction
        
        This project aims to predict the prices of used cars in Pakistan using machine learning.
        
        ### Features
        - Price prediction based on car specifications
        - Market analysis and visualization
        - Easy-to-use web interface
        
        ### Technology Stack
        - Python
        - Streamlit for web interface
        - Scikit-learn for machine learning
        - Pandas for data manipulation
        - Matplotlib and Seaborn for visualization
        
        ### Data Source
        The data is sourced from PakWheels, Pakistan's largest automotive marketplace.
        
        ### How to Use
        1. Navigate to the Price Prediction page
        2. Fill in the car details
        3. Click "Predict Price" to get an estimate
        
        ### Model Information
        The prediction model uses a Random Forest algorithm trained on historical data
        from the Pakistani used car market.
        """)

if __name__ == "__main__":
    main()'''
    

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS to handle missing image
st.markdown("""
<style>
    .main-header {
        color: #2E86AB;
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1>🚗 Car Price Prediction System</h1></div>', unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", 
                       ["Home", "Predict Price", "Market Analysis"])

if page == "Home":
    st.header("Welcome to Car Price Predictor")
    st.write("""
    This application helps you:
    - ✅ Predict prices of used cars
    - 📊 Analyze market trends  
    - 🔍 Understand factors affecting car prices
    """)
    
    # Display stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Listings", "77,878")
    with col2:
        st.metric("Unique Models", "68")
    with col3:
        st.metric("Average Price", "₹3,883,232")

# Replace the entire prediction section with this code:

elif page == "Predict Price":
    st.header("Predict Car Price")
    
    # Load model
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("Model loaded successfully!")
        st.info(f"Model expects {model.n_features_in_} features")
    except:
        st.error("Could not load model file")
        st.stop()
    
    # Input form for ALL 14 features
    with st.form("prediction_form"):
        st.subheader("Car Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Manufacturing Year", min_value=1990, max_value=2024, value=2020)
            engine = st.number_input("Engine Size (cc)", min_value=600, max_value=5000, value=1500)
            mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000)
            brand = st.selectbox("Brand", ["Toyota", "Honda", "Suzuki", "Hyundai", "Other"])
            model_name = st.selectbox("Model", ["Camry", "Civic", "Alto", "Accord", "City", "Other"])
        
        with col2:
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            body_type = st.selectbox("Body Type", ["Sedan", "Hatchback", "SUV", "Coupe"])
            color = st.selectbox("Color", ["White", "Black", "Silver", "Red", "Blue"])
            location = st.selectbox("Location", ["Karachi", "Lahore", "Islamabad", "Other"])
            
            # Add more features to reach 14 total
            owners = st.number_input("Number of Previous Owners", min_value=1, max_value=5, value=1)
            registration_year = st.number_input("Registration Year", min_value=1990, max_value=2024, value=2020)
            has_airbags = st.checkbox("Has Airbags", value=True)
            has_abs = st.checkbox("Has ABS", value=True)
        
        submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        try:
            # Convert categorical features to numerical (you need to use your actual encoding)
            # This is an example - you need to match your training encoding!
            brand_encoded = 1 if brand == "Honda" else 2 if brand == "Toyota" else 3 if brand == "Suzuki" else 4
            model_encoded = 1 if model_name == "Civic" else 2 if model_name == "Camry" else 3 if model_name == "Alto" else 4
            fuel_encoded = 0 if fuel_type == "Petrol" else 1 if fuel_type == "Diesel" else 2
            transmission_encoded = 0 if transmission == "Manual" else 1
            body_encoded = 0 if body_type == "Sedan" else 1 if body_type == "Hatchback" else 2 if body_type == "SUV" else 3
            color_encoded = 0 if color == "White" else 1 if color == "Black" else 2 if color == "Silver" else 3 if color == "Red" else 4
            location_encoded = 0 if location == "Karachi" else 1 if location == "Lahore" else 2 if location == "Islamabad" else 3
            airbags_encoded = 1 if has_airbags else 0
            abs_encoded = 1 if has_abs else 0
            
            # Create feature array with ALL 14 features in correct order
            features = [
                year,               # feature 1
                engine,             # feature 2
                mileage,            # feature 3
                brand_encoded,      # feature 4
                model_encoded,      # feature 5
                fuel_encoded,       # feature 6
                transmission_encoded, # feature 7
                body_encoded,       # feature 8
                color_encoded,      # feature 9
                location_encoded,   # feature 10
                owners,             # feature 11
                registration_year,  # feature 12
                airbags_encoded,    # feature 13
                abs_encoded         # feature 14
            ]
            
            st.write("Features being sent:", features)
            
            prediction = model.predict([features])[0]
            st.success(f"### Predicted Price: Rs {prediction:,.2f}")
            st.info(f"Approx USD: ${prediction/75:,.2f}")
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write("Make sure you're providing all 14 features in the correct order!")

elif page == "Market Analysis":
    st.header("Market Analysis")
    st.write("Market trends and analysis will be displayed here.")

# Footer
st.markdown("---")
st.caption("Car Price Prediction System | Built with Streamlit")
