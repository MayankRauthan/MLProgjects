# loan_prediction_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

st.title("Loan Status Prediction")

# Upload the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    loan_dataset = pd.read_csv(uploaded_file)

    # Display the first 5 rows of the dataframe
    st.write("First 5 rows of the dataset:")
    st.write(loan_dataset.head())

    # Number of rows and columns
    st.write("Dataset shape:", loan_dataset.shape)

    # Statistical measures
    st.write("Statistical measures:")
    st.write(loan_dataset.describe())

    # Number of missing values in each column
    st.write("Missing values in each column:")
    st.write(loan_dataset.isnull().sum())

    # Dropping the missing values
    loan_dataset = loan_dataset.dropna()

    # Number of missing values in each column after dropping missing values
    st.write("Missing values after dropping:")
    st.write(loan_dataset.isnull().sum())

    # Label encoding
    loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

    # Display the first 5 rows of the dataframe after encoding
    st.write("First 5 rows after encoding:")
    st.write(loan_dataset.head())

    # Replacing the value of 3+ to 4
    loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

    # Data visualization
    st.write("Education & Loan Status:")
    sns_plot = sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
    st.pyplot(sns_plot.figure)

    st.write("Marital Status & Loan Status:")
    sns_plot = sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
    st.pyplot(sns_plot.figure)

    # Convert categorical columns to numerical values
    loan_dataset.replace({'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, 'Female': 0},
                          'Self_Employed': {'No': 0, 'Yes': 1}, 'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                          'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

    # Separating the data and label
    X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
    Y = loan_dataset['Loan_Status']

    st.write("Feature set:")
    st.write(X.head())
    st.write("Labels:")
    st.write(Y.head())

    # Train Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

    st.write("Training set shape:", X_train.shape)
    st.write("Test set shape:", X_test.shape)

    # Training the Support Vector Machine Model
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)

    # Model Evaluation
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    st.write('Accuracy on training data:', training_data_accuracy)

    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    st.write('Accuracy on test data:', test_data_accuracy)

    # Making a predictive system
    st.header("Make a Prediction")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income")
    coapplicant_income = st.number_input("Coapplicant Income")
    loan_amount = st.number_input("Loan Amount")
    loan_amount_term = st.number_input("Loan Amount Term")
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    # Convert categorical inputs to numerical values
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0
    property_area = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    dependents = 4 if dependents == "3+" else int(dependents)

    # Prepare the feature vector
    features = np.array([[gender, married, dependents, education, self_employed, applicant_income,
                          coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area]])

    if st.button("Predict Loan Status"):
        prediction = classifier.predict(features)
        result = "Approved" if prediction[0] == 1 else "Rejected"
        st.write(f"Loan Status: {result}")
