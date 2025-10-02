import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf

# ----------------------------
# Load Model & Encoders
# ----------------------------
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #c3ec52 0%, #0ba29d 100%);
    }
    .main {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.25);
    }
    .big-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #0ba29d;
        text-align: center;
    }
    .prediction-box {
        background-color: #f9f9f9;
        border: 2px solid #0ba29d;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 14px;
        color: #333;
    }
    .stButton>button {
        background-color: #0ba29d;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        font-weight: bold;
        height: 3em;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #097b77;
        color: #fff;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Title Section
# ----------------------------
st.markdown("<p class='big-font'>💳 Customer Churn Prediction using ANN</p>", unsafe_allow_html=True)
st.markdown("### Predict whether a bank customer is likely to churn using an **Artificial Neural Network (ANN)**")
st.write("---")

# ----------------------------
# Sidebar Input
# ----------------------------
st.sidebar.header("📌 Enter Customer Details")

geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('🎂 Age',18,92,40)
balance = st.sidebar.number_input('💰 Balance', value=60000.0)
credit_score = st.sidebar.number_input('💳 Credit Score', value=600)
estimated_salary = st.sidebar.number_input('💵 Estimated Salary', value=50000.0)
tenure = st.sidebar.slider('📅 Tenure (Years with Bank)', 0, 10, 3)
num_of_products = st.sidebar.slider('🛍️ Number of Products',1 , 4,2)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card?', [0,1])
is_active_member = st.sidebar.selectbox('✅ Active Member?', [0,1])

# ----------------------------
# Process Input
# ----------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("🚀 Predict Churn"):
    prediction = model.predict(input_data_scaled)
    prediction_prob = prediction[0][0]

    if prediction_prob > 0.5:
        st.markdown(f"<div class='prediction-box' style='color:red;'>⚠️ The customer is LIKELY to churn.<br>Probability: {prediction_prob:.2f}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-box' style='color:green;'>✅ The customer is NOT likely to churn.<br>Probability: {prediction_prob:.2f}</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<div class='footer'>✨ Built with ❤️ using Streamlit & ANN<br>Developed by <b>Mohammed Yousuf</b></div>", unsafe_allow_html=True)
