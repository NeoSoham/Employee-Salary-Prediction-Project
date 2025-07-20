🚀 EMPLOYEE SALARY PREDICTION

🎯 A machine learning web app to predict whether an employee earns >50K or ≤50K based on personal and professional attributes. Built using Streamlit, trained on the UCI Adult Income Dataset.

📦 PROJECT OVERVIEW

🔍 This project aims to:

•	Classify employee salary range (>50K or ≤50K)
•	Leverage preprocessing techniques (Encoding, Scaling, PCA)
•	Utilize a trained machine learning model
•	Provide an interactive web interface using Streamlit
•	Visualize feature importance and model performance

🗂️ PROJECT STRUCTURE

Employee Salary Prediction Project/

•	📄 app.py → Main Streamlit application
•	📄 Employee_Salary_Prediction.ipynb → Jupyter notebook for EDA & model building
•	📦 best_model.pkl → Trained machine learning model
•	📦 scaler.pkl → Scaler for numerical features
•	📦 pca.pkl → PCA transformer
•	📦 label_encoders.pkl → Encoded categorical variables
•	📄 feature_importance.py → Script to plot feature importance
•	📄 visualizations.py → Custom visualization utilities
•	🗃️ .git/ → Git version control metadata

⚙️ HOW TO RUN LOCALLY

✅ PREREQUISITES

•	Python 3.8 or higher
•	Required libraries: streamlit, pandas, numpy, joblib, plotly, scikit-learn

Install all dependencies using:

pip install -r requirements.txt

▶️ TO START THE STREAMLIT APP

streamlit run app.py

📊 MODEL INFORMATION

•	Dataset: UCI Adult Income
•	Problem Type: Binary Classification
•	Target Variable: Salary (<=50K or >50K)
•	Preprocessing:
o	Label Encoding for categorical columns
o	StandardScaler for numerical features
o	PCA for dimensionality reduction
•	Model: Final model saved as best_model.pkl after evaluation
•	Evaluation Metrics: Accuracy, F1-score, Precision, Recall

