# 🏡 Superhost Predictor – Airbnb Classification with Logistic Regression
This project is part of a Machine Learning Foundations lab focused on the ML lifecycle, particularly the evaluation and deployment phases. The goal is to train a logistic regression model to predict whether an Airbnb host is a superhost based on listing features. The final model is optimized, visualized, and persisted for future use.

📌 Project Objectives
Frame a binary classification problem using preprocessed Airbnb listing data.

Train, test, and evaluate logistic regression models using scikit-learn.

Perform hyperparameter tuning with GridSearchCV to optimize model performance.

Generate and interpret evaluation metrics: confusion matrix, ROC curve, AUC, and precision-recall curves.

Use feature selection to identify the top predictors of superhost status.

Persist the final model using Python’s pickle module.

Upload the trained model and dataset to GitHub for reproducibility.

⚙️ Tech Stack
Python 3

Pandas – data manipulation

NumPy – numerical operations

Matplotlib & Seaborn – visualizations

scikit-learn – modeling, evaluation, feature selection, tuning

Pickle – model serialization

Git & GitHub – version control and collaboration

📁 Dataset
File: airbnbData_train.csv

The dataset includes preprocessed features such as:

One-hot encoded categorical variables

Scaled numerical features

Imputed missing values

Label: host_is_superhost (True/False)

📈 Model Workflow
Load Data
pd.read_csv() from data_LR/airbnbData_train.csv

Data Splitting
train_test_split with 90% training, 10% testing

Model Training

Default Logistic Regression (C=1.0)

GridSearchCV to tune hyperparameter C

Evaluation

Confusion Matrix

ROC & Precision-Recall Curves

AUC Comparison

Feature Selection with SelectKBest

Model Deployment

Saved final model using pickle as Pickle_Logistic_Regression.pkl

🔍 Results
Best AUC: ~0.823 using 10 selected features

Top Predictive Features:

host_response_rate, host_acceptance_rate

review_scores_rating, review_scores_value

reviews_per_month, and others

📦 How to Use
bash
Copy
Edit
# Clone this repo
git clone https://github.com/Hrios05/MachineLearningFoundations_Lab.git

# Navigate to the project directory
cd MachineLearningFoundations_Lab

# Load the model in your Python script or notebook
import pickle
model = pickle.load(open('Pickle_Logistic_Regression.pkl', 'rb'))

# Use the model to predict on new data
predictions = model.predict(X_new)
🧠 Learnings
Importance of hyperparameter tuning in classification tasks

How model evaluation metrics provide different insights

Practice in deploying and persisting ML models for future inference

