# SalaryPredictor_MLApp_SLR
This repository contains a Simple Linear Regression model implemented in Python to predict employee salaries based on years of experience.

# Salary Predictor App

This repository contains a **Salary Predictor App** that uses a Simple Linear Regression model to predict employee salaries based on their years of experience. The project includes both a backend for training the model and a frontend built with Streamlit for user interaction.

## Project Structure

- `app.py`: The Streamlit application for user input and salary prediction.
- `Salary_Data.csv`: The dataset containing employee salaries and years of experience.
- `linear_regression_model.pkl`: The trained linear regression model saved for predictions.
- `backend.py`: The script that trains the linear regression model.

## Getting Started

### Prerequisites

Make sure you have Python installed on your machine. You will also need to install the required libraries. You can do this by running:

```bash
pip install pandas numpy scikit-learn matplotlib streamlit 
Running the App
Clone the repository to your local machine:

bash
Copy code
git clone <your-repo-url>
cd <your-repo-name>
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Open the provided URL in your web browser to access the app.

How to Use
Enter the number of years of experience in the input field.
Click the "Predict Salary" button.
The app will display the predicted salary based on the input.
