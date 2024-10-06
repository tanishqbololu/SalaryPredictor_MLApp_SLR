import streamlit as st 
import pickle
import numpy as np


# Load saved model
model = pickle.load(open(r"D:\A VS CODE\2. ML\linear_regression_model.pkl", 'rb'))

# Set the title of Streamlit app
st.title("Salary Prediction App")

# Add a breif description
st.write("This app predicts salary based on years of experience using a simple regression model.")

# Add input widget for user to enter years of experience
years_experience = st.number_input("Enter Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

#When the button is clicked, make predictions
if st.button("Predict Salary"):
    # Make a prediction using the trained model
    experience_input = np.array([[years_experience]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(experience_input)
   
    # Display the result
    st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.")

