# 🧠 Customer Churn Prediction using ANN

This project predicts whether a bank customer will **churn** (leave the bank) using an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras**.  

It covers the complete lifecycle:
- **Data Preprocessing** (encoding, scaling)
- **ANN Model Training**
- **Model Inference**
- **Streamlit App Deployment**

---

## 📌 Key Features
✅ Preprocessing pipeline with **LabelEncoder**, **OneHotEncoder**, and **StandardScaler**  
✅ **ANN Implementation** with TensorFlow/Keras  
✅ EarlyStopping & TensorBoard for monitoring  
✅ Saved model & encoders (`.h5` and `.pkl` files)  
✅ **Streamlit Web App** for interactive predictions  

---

## 🏗️ ANN Architecture

The ANN used in this project is a **feedforward neural network**:

- **Input Layer** → 11 features after encoding  
- **Hidden Layer 1** → 64 neurons, ReLU activation  
- **Hidden Layer 2** → 32 neurons, ReLU activation  
- **Output Layer** → 1 neuron, Sigmoid activation  

### Model Summary
```
Dense(64, activation='relu', input_shape=(X_train.shape[1],))
Dense(32, activation='relu')
Dense(1, activation='sigmoid')
```

### Optimizer & Loss
- Optimizer: **Adam** (lr=0.01)  
- Loss: **Binary Crossentropy**  
- Metric: **Accuracy**  

---

## 📂 Project Structure
```
├── experiment.ipynb          # Training with ANN
├── prediction.ipynb          # Testing / inference
├── app.py                    # Streamlit app for predictions
├── model.h5                  # Trained ANN model
├── scaler.pkl                # StandardScaler object
├── label_encoder_gender.pkl  # LabelEncoder for Gender
├── onehot_encoder_geo.pkl    # OneHotEncoder for Geography
├── Churn_Modelling.csv       # Dataset
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/customer-churn-ann.git
cd customer-churn-ann
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Training the Model
Run **experiment.ipynb** to:
1. Preprocess the dataset  
2. Train the ANN  
3. Monitor training with TensorBoard  
4. Save the trained model and encoders  

---

## 🔮 Making Predictions
Use **prediction.ipynb** to:
- Load the saved ANN model & encoders  
- Pass new customer details  
- Get churn probability  

---

## 🌐 Streamlit App
Run the app for interactive predictions:
```bash
streamlit run app.py
```

### App Features:
- Enter customer details (age, salary, balance, geography, etc.)  
- ANN predicts churn probability  
- Displays whether customer is **likely to churn** or **not likely to churn**  

---

## 📊 Dataset
**Churn_Modelling.csv**  
Includes 10,000 bank customers with features like:
- CreditScore  
- Geography  
- Gender  
- Age  
- Balance  
- Tenure  
- Products  
- EstimatedSalary  
- Exited (target: 1 = churned, 0 = retained)  

---

## 🚀 Future Enhancements
- Hyperparameter tuning for ANN  
- Comparison with other ML models (Logistic Regression, XGBoost)  
- Deploy app on **Streamlit Cloud / Heroku / AWS**  
- Add **Explainability (SHAP/LIME)**  

---

## 👨‍💻 Author
Developed by Mohammed Yousuf

If you find this useful, don’t forget to ⭐ the repo!
