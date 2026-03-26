import streamlit as st
import pandas as pd
import joblib

# load model and encoders
model = joblib.load("loan_model.pkl")
le = joblib.load("label_encoder.pkl")
on = joblib.load("onehot_encoder.pkl")
cols = joblib.load("model_columns.pkl")

st.title("Loan Approval Predictor")

age = st.number_input("Age",18,80)
income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.slider("Credit Score",300,900)

education = st.selectbox("Education Level",["Graduate","Not Graduate"])
employment = st.selectbox("Employment Status",["Salaried","Self-Employed","Unemployed"])
gender = st.selectbox("Gender",["Male","Female"])
marital = st.selectbox("Marital Status",["Married","Single"])

dependents = st.number_input("Dependents")
dti = st.number_input("DTI Ratio")
existing_loans = st.number_input("Existing Loans")
savings = st.number_input("Savings")
collateral = st.number_input("Collateral Value")

purpose = st.selectbox("Loan Purpose",["Personal","Car","Business","Home","Education"])
term = st.number_input("Loan Term")

property_area = st.selectbox("Property Area",["Urban","Semiurban","Rural"])
employer = st.selectbox("Employer Category",["Private","Government","Unemployed","MNC","Business"])


if st.button("Predict"):

    df = pd.DataFrame([{
        "Age":age,
        "Applicant_Income":income,
        "Coapplicant_Income":co_income,
        "Loan_Amount":loan_amount,
        "Credit_Score":credit_score,
        "Education_Level":education,
        "Employment_Status":employment,
        "Gender":gender,
        "Marital_Status":marital,
        "Dependents":dependents,
        "DTI_Ratio":dti,
        "Existing_Loans":existing_loans,
        "Savings":savings,
        "Collateral_Value":collateral,
        "Loan_Purpose":purpose,
        "Loan_Term":term,
        "Property_Area":property_area,
        "Employer_Category":employer
    }])

    # convert Graduate / Not Graduate → Yes / No
    education_value = "Yes" if education == "Graduate" else "No"

# apply label encoder
    df["Education_Level"] = le.transform([education_value])[0]

    # one hot encode
    cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
                "Property_Area","Gender","Employer_Category"]

    encoded = on.transform(df[cat_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=on.get_feature_names_out(cat_cols)
    )

    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
    df = df.reindex(columns=cols, fill_value=0)


    prediction = model.predict(df)[0]

    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")