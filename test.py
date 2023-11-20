import streamlit as st
import joblib
import numpy as np

# Load the trained model
lasso_model = joblib.load('lasso_model.pkl')

# Load the scaler
scaler = joblib.load('scaler_model.pkl')

def main():
    st.title("Your Model Deployment")

    # Collect user input
    user_input = get_user_input()

    # Preprocess the user input
    processed_input = preprocess_input(user_input)

    # Make predictions
    prediction = predict(processed_input)

    # Display the prediction
    st.write(f"Prediction: {prediction}")

def get_user_input():
    # Add widgets to collect user input
    # For example, if your features are numeric:
    feature1 = st.slider("Feature 1", min_value=60, max_value=100)
    # feature2 = st.slider("Feature 2", min_value=..., max_value=...)

    # Return user input as a dictionary
    return {'feature1': feature1}

def preprocess_input(user_input):
    # Standardize the user input using the loaded scaler
    # Convert user input to a numpy array before standardization
    input_array = np.array(list(user_input.values())).reshape(1, -1)
    standardized_input = scaler.transform(input_array)

    return standardized_input

def predict(processed_input):
    # Make predictions using the loaded model
    prediction = lasso_model.predict(processed_input)

    # You may need to inverse_transform if your output is standardized
    # prediction = scaler.inverse_transform(prediction.reshape(1, -1))

    return prediction[0]

if __name__ == "__main__":
    main()
