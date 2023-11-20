from string import punctuation
import pandas as pd
import re
import streamlit as st
import joblib

# Import the necessary libraries for preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the random forest model and scaler
model_rf = joblib.load('random_forest.pkl')
scaler = joblib.load('standard_scaler.pkl')

# Define the list of numerical variables
numerical_vars = ['RETIREMENT_AGE', 'RETIREMENT_FUND_VALUE', 'DEPT_VALUE',
       'SPARE_CASH_VALUE', 'OTHER_MONTHLY_SUPPORTING_VALUE',
       'CRITICAL_ILLNESS', 'SPOUSE_GENDER', 'SPOUSE_RETIREMENT_AGE',
       'SPOUSE_DATE_OF_BIRTH', 'INTERNATIONAL_CASH_UNIT_TRUST',
       'SA_EQUITY_LAP', 'SA_BOND_LAP', 'SA_CASH_LAP', 'INTERNATIONAL_CASH_LAP',
       'LA_EAC_PA_INCL_VAT', 'UNIT_TRUST_EAC_PA_INCL_VAT']

# Define the list of categorical variables
categorical_vars = ['CRITICAL_ILLNESS', 'SPOUSE_GENDER']

# Create a label encoder object
label_encoder = LabelEncoder()

def preprocess_input(input_data):
    # Convert date of birth to year
    if 'SPOUSE_DATE_OF_BIRTH' in input_data.columns:
        input_data['SPOUSE_DATE_OF_BIRTH'] = pd.to_datetime(input_data['SPOUSE_DATE_OF_BIRTH']).dt.year.astype('Int64')

    # Label encode other categorical variables
    for col in categorical_vars:
        input_data[col] = label_encoder.fit_transform(input_data[col].astype(str))

    # Standardize numerical features
    input_data[numerical_vars] = scaler.transform(input_data[numerical_vars])

    return input_data

def predict_retirement_income(input_data):
    # Make prediction
    prediction = model_rf.predict(input_data.values.reshape(1, -1))

    # Inverse transform to get the original scale
    prediction = scaler.inverse_transform(input_data.values.reshape(1, -1))

    return prediction

def main():
    st.title('Retirement Income Estimator')

    # Get user input
    retirement_age = st.slider('Enter Retirement Age:', min_value=50, max_value=100, value=65)
    retirement_fund_value = st.number_input('Enter Retirement Fund Value:')
    dept_value = st.number_input('Enter Debt Value:')
    spare_cash_value = st.number_input('Enter Spare Cash Value:')
    other_monthly_supporting_value = st.number_input('Enter Other Monthly Supporting Value:')
    critical_illness = st.selectbox('Select Critical Illness:', ['Yes', 'No'])
    spouse_gender = st.selectbox('Select Spouse Gender:', ['Male', 'Female', 'NAN'])
    spouse_retirement_age = st.number_input('Enter Spouse Retirement Age:')
    spouse_date_of_birth = st.date_input('Enter Spouse Date of Birth:', None)  # Default to None 
    international_cash_unit_trust = st.number_input('Enter International Cash Unit Trust:')
    sa_equity_lap = st.number_input('Enter SA Equity LAP:')
    sa_bond_lap=st.number_input('Enter SA bond LAP: ')
    sa_cash_lap=st.number_input('Enter SA cash LAP: ')
    international_cash_lap=st.number_input('Enter International cash LAP: ')
    la_eac_pa_incl_vat=st.number_input('Enter la eac pa incl vat: ')
    unit_trust_eac_pa_incl_vat=st.number_input('Enter unit trust eac pa incl vat: ')

    if st.button('Submit'):
        # Create a DataFrame with user input
        user_input = pd.DataFrame({
            'RETIREMENT_AGE': [retirement_age],
            'RETIREMENT_FUND_VALUE': [retirement_fund_value],
            'DEPT_VALUE': [dept_value],
            'SPARE_CASH_VALUE': [spare_cash_value],
            'OTHER_MONTHLY_SUPPORTING_VALUE': [other_monthly_supporting_value],
            'CRITICAL_ILLNESS': [critical_illness],
            'SPOUSE_GENDER': [spouse_gender],
            'SPOUSE_RETIREMENT_AGE': [spouse_retirement_age],
            'SPOUSE_DATE_OF_BIRTH': [spouse_date_of_birth],
            'INTERNATIONAL_CASH_UNIT_TRUST': [international_cash_unit_trust],
            'SA_EQUITY_LAP': [sa_equity_lap],
            'SA_BOND_LAP':[sa_bond_lap],
            'SA_CASH_LAP':[sa_cash_lap],
            'INTERNATIONAL_CASH_LAP':[international_cash_lap],
            'LA_EAC_PA_INCL_VAT':[la_eac_pa_incl_vat],
            'UNIT_TRUST_EAC_PA_INCL_VAT':[unit_trust_eac_pa_incl_vat]
        })

        # Preprocess input data
        input_data = preprocess_input(user_input)

        # Make predictions and inverse transform
        predicted_income = predict_retirement_income(input_data)

        # Display the prediction
        st.header('Retirement Income Prediction:')
        st.write(f'The estimated retirement income is: ${predicted_income:,.2f}')

if __name__ == '__main__':
    main()



