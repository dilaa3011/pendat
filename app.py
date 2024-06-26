import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score

# Set page config
st.set_page_config(page_title="Pendat Personal", layout="wide")

# Load dataset
df_data = pd.read_excel('balloons.xlsx')

# Title and description
st.title("Pendat Personal - Balloon Prediction")
st.write("Predict whether a balloon will be inflated based on selected features.")

# Sidebar for feature inputs
st.sidebar.header("Input Features")

# Input select boxes for each feature with default option
selected_color = st.sidebar.selectbox('Select Color:', [''] + list(df_data['color'].unique()), key='color')
selected_size = st.sidebar.selectbox('Select Size:', [''] + list(df_data['size'].unique()), key='size')
selected_act = st.sidebar.selectbox('Select Act:', [''] + list(df_data['act'].unique()), key='act')
selected_age = st.sidebar.selectbox('Select Age:', [''] + list(df_data['age'].unique()), key='age')

# Check if all inputs are selected
if selected_color and selected_size and selected_act and selected_age:
    # Capture user selections in new_data
    new_data = {
        'color': [selected_color],
        'size': [selected_size],
        'act': [selected_act],
        'age': [selected_age]
    }

    # Create DataFrame from new data
    df_new = pd.DataFrame(new_data)

    # Encode categorical columns using LabelEncoder
    label_encoders = {}
    categorical_columns = ['color', 'size', 'act', 'age']

    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        df_data[col] = label_encoders[col].fit_transform(df_data[col])  # Ensure encoding consistency
        df_new[col] = label_encoders[col].transform(df_new[col])  # Transform df_new

    # Separate features and target from entire dataset
    X = df_data[['color', 'size', 'act', 'age']]
    y = df_data['inflated']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model_rfc = RandomForestClassifier()
    model_rfc.fit(X_train, y_train)  # Train model on training set

    # Predict using the trained model on df_new
    new_predictions = model_rfc.predict(df_new)

    # Map predicted values to 'T' or 'F'
    df_new['Predicted'] = new_predictions
    df_new['Predicted'] = df_new['Predicted']

    # Display prediction result
    st.subheader("Prediction Result")
    st.write(df_new[['color', 'size', 'act', 'age', 'Predicted']])

    # Evaluate model on test set
    y_pred = model_rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

else:
    st.sidebar.write("Please select values for all features.")
