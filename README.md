# Credit Default Risk Predictor

A robust, end-to-end machine learning pipeline and web application designed to evaluate and predict the likelihood of credit card default. This tool leverages financial data to provide an accurate risk assessment, utilizing a modular architecture, containerization, and automated testing for production readiness.

## Overview

Financial institutions face significant challenges in accurately assessing credit risk. This project provides a reliable, data-driven solution by training classification models and deploying the best-performing pipeline via a RESTful Flask API. The architecture is modularized into distinct components for data processing, feature engineering, model training, and API serving, ensuring scalability and maintainability.

## Key Features

End-to-End ML Pipeline: Modular source code (`source/`) handling everything from data ingestion to feature engineering and model training.
API Backend: A Flask application (`api/app.py`) serving predictions via a dedicated endpoint.
Production-Ready Engineering: Includes a `Dockerfile` for containerized deployment and a `Makefile` for streamlined build automation.
Experiment Tracking & Logging: Structured logging (`logs/project.log`) to monitor pipeline execution and API health.

# Tech Stack

Machine Learning & Data Processing:** Python, Scikit-Learn, Pandas, NumPy
Backend Framework:** Flask
Deployment & DevOps: Docker, Makefile
Environment Management: `python-dotenv`, Virtual Environment (`ml3env`)

# Project Structure

CREDITDEFAULTRISKPREDICTOR/
├── .github/                  # CI/CD workflows
├── api/                      # Flask application and frontend
│   ├── templates/            # HTML frontend interfaces
│   └── app.py                # Main Flask API script
├── data/                     # Data storage (ignored in version control)
│   ├── processed/            # Cleaned, ready-to-use data
│   ├── raw/                  # Original, unmodified data
│   ├── raw_csv/              # Raw CSV extracts
│   └── README.md             # Data dictionary and details
├── logs/                     # Application and training logs
│   └── project.log           
├── models/                   # Serialized model artifacts
│   └── best_model_pipeline.pkl
├── source/                   # Core machine learning pipeline modules
│   ├── data/                 # Data ingestion scripts
│   ├── features/             # Feature engineering scripts
│   ├── models/               # Model training and evaluation scripts
│   └── utils/                # Helper functions and utilities
├── .env                      # Environment variables configuration
├── .gitignore                # Files to ignore in Git
├── Dockerfile                # Instructions for containerizing the app
├── LICENSE                   # Open-source license
├── Makefile                  # Automated commands for setup, testing, and running
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies


# Installation and Setup

1. Clone the repository

git clone https://github.com/Mew-Mew1/credit-default-risk-predictor.git
cd credit-default-risk-predictor


2. Environment Setup
Create and activate a virtual environment (named `ml3env` as per the project structure):

python -m venv ml3env
source ml3env/bin/activate  # On Windows use: ml3env\Scripts\activate


3. Install Dependencies

pip install -r requirements.txt


# Usage

# Running Locally with Python

Start the Flask application:
python api/app.py

Access the application by navigating to `http://127.0.0.1:5000` in your browser.

# Running via Docker

Build the Docker image:
docker build -t credit-risk-predictor .

Run the container:
docker run -p 5000:5000 credit-risk-predictor

