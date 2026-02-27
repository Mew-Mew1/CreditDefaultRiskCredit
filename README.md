# Credit Default Risk Predictor

A robust, end-to-end machine learning pipeline and web application designed to evaluate and predict the likelihood of credit card default. This tool leverages the Default of Credit Card Clients Dataset to provide a 3-tier risk assessment (Minimal, Moderate, Critical) based on user financial profiles.

## Overview

Financial institutions face significant challenges in accurately assessing credit risk. This project provides a reliable, data-driven solution by training multiple classification models and deploying the best-performing one (based on ROC-AUC score) via a RESTful Flask API. The user-friendly web interface allows for seamless input of 11 key financial features to receive immediate, professional-grade risk evaluations.

# Key Features

Multi-Model ML Pipeline: Automates the training, hyperparameter tuning, and evaluation of Logistic Regression, Random Forest, and XGBoost models.
Optimal Model Selection: Dynamically selects and saves the model with the highest ROC-AUC score for inference.
API: A Flask-based backend featuring a `/predict` endpoint that securely processes incoming JSON requests and serves predictions.
Tiered Risk Assessment: Outputs predictions in clear, financial terminology:
🟢 Minimal Risk
🟡 Moderate Risk
🔴 Critical Risk



# Tech Stack

Machine Learning: Python, Scikit-Learn, XGBoost, Pandas, NumPy
Backend: Flask, Pickle
Frontend: Used AI for the same
Data Source: Default of Credit Card Clients Dataset

# Project Structure

credit-default-risk-predictor/
├── app.py                 # Flask application and API routes
├── model_training.py      # ML pipeline, tuning, and model export script
├── requirements.txt       # Python dependencies
├── best_model.pkl         # Serialized optimal ML model
├── templates/
│   └── index.html         # Styled frontend interface
└── README.md              # Project documentation

```

# Installation and Setup

1. Clone the repository

git clone https://github.com//credit-default-risk-predictor.git
cd credit-default-risk-predictor

2. Create a virtual environment

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Train the models (Optional)
If you want to retrain the models on the dataset to generate a new best_model.pkl.

python model_training.py

5. Run the Flask application

python app.py


6. Access the application
Open your web browser and navigate to `http://127.0.0.1:5000` to interact with the predictor.

Usage
1. Launch the web application.
2. Enter the required 11 financial metrics (e.g., credit limit, payment history, bill amounts) into the provided form fields.
3. Click the "Predict Risk" button.
4. The system will process the data through the Flask `/predict` endpoint and display the resulting risk tier instantly.