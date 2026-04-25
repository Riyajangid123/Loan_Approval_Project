## Loan Approval Prediction Project
# Overview
 This project predicts whether a loan application will be approved based on applicant data. 
 It uses a machine learning pipeline for preprocessing, feature engineering, and modeling. 
 The project is deployed as a web application using FastAPI for the backend and Streamlit for the frontend, 
 packaged in Docker for easy deployment.

# Features
- Data Preprocessing & EDA
- Handled missing values and outliers
- Explored feature distributions and correlations
- Visualized trends and patterns to understand data

# Machine Learning Pipeline
- Utilized ColumnTransformer to handle numerical and categorical features separately
- Scaled numerical features and encoded categorical features efficiently
- Model trained with Logistic Regression (or chosen ML model) for classification

# Model Performance
- Accuracy: 0.7886
- Cross-validation Score: 0.8013
- Proper evaluation metrics ensure reliable predictions

# Deployment
- Backend: FastAPI API serving predictions
- Frontend: Streamlit interface for user input
- Model Persistence: Saved using Joblib
- Containerization: Dockerized for easy deployment on cloud platforms
- Cloud Deployment: Successfully deployed on Amazon Web Services (EC2) with Docker containers, enabling scalable and accessible web-based predictions.

# Technologies Used
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn (EDA & Visualization)
- Scikit-learn (Preprocessing, ColumnTransformer, Modeling, Evaluation)
- FastAPI (API Deployment)
- Streamlit (Web App Interface)
- Joblib (Model Serialization)
- Docker (Containerization & Deployment)
- AWS EC2

# Project Structure
loan-approval-project/
│
├── app/
│   ├── main.py             # FastAPI backend
│   ├── streamlit_app.py    # Streamlit frontend
│
├── models/
│   └── loan_model.pkl      # Trained ML model
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb  # Exploratory Data Analysis & Pipeline
│
├── Dockerfile              # Docker container configuration
├── requirements.txt        # Project dependencies
└── README.md               # Project overview & instructions
