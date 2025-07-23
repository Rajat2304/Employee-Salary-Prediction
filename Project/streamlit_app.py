import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_salary_model.pkl")

st.title("💼Employee  Salary Predicton  ")
st.write("Enter employee details below for salary prediction, or use batch CSV upload.")

education_levels = ['HS-grad', 'Some-college', 'Assoc', 'Bachelors', 'Masters', 'PhD']
occupation_dict = {
    'HS-grad': ['Sales Associate','Marketing Manager','Graphic Designer'],
    'Some-college': [
        'Tech-support', 'Craft-repair', 'Other-service', 'Handlers-cleaners',
        'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
        'Priv-house-serv', 'Protective-serv', 'Armed-Forces'
    ],
    'Assoc': ['Quality Analyst','Plant Supervisor','Nurse'],
    'Bachelors': ['Software Engineer','Business Analyst','Sales Associate'],
    'Masters': ['Data Scientist','Project Manager','Accountant'],
    'PhD': ['Professor','Research Scientist','Legal Advisor']
}

st.sidebar.header("Input Employee Details")
education = st.sidebar.selectbox("Education", education_levels)
occupation_options = occupation_dict.get(education, occupation_dict['HS-grad'])
occupation = st.sidebar.selectbox("Occupation", occupation_options)
age = st.sidebar.slider("Age", 18, 65, 30)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
hours_per_week = st.sidebar.slider("Hours per Week", 20, 80, 40)

input_df = pd.DataFrame([{
    'age': age,
    'education': education,
    'occupation': occupation,
    'experience': experience,
    'hours-per-week': hours_per_week
}])

st.write("#### Selected Profile Input")
st.write(input_df)

if st.button("Predict Salary"):
    pred = model.predict(input_df)[0]
    st.success(f"🤑 Predicted Salary: ${int(pred):,}")

# Batch prediction support
st.markdown("---")
st.subheader("Batch Prediction from CSV")
csv_file = st.file_uploader("Upload CSV file", type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file)
    st.write("Preview of Uploaded Data:", df.head())
    preds = model.predict(df)
    df['Predicted_Salary'] = preds.astype(int)
    st.write("Predictions:", df.head())
    csv_pred = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predicted CSV", csv_pred, "salary_predictions.csv","text/csv")
