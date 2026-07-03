# 🏦 Bank Customer Churn Prediction using Artificial Neural Networks (ANN)

> Predict whether a bank customer is likely to leave the bank using an Artificial Neural Network built with TensorFlow/Keras and deployed as an interactive Streamlit web application.

## 🚀 Live Demo

🌐 **Live App:**
https://customer-churn-prediction-ann-vgdrudtkydmowwv9ev9noi.streamlit.app/

💻 **GitHub Repository:**
https://github.com/niraj-codes01/customer-churn-prediction-ann

---

## 📌 Project Overview

Customer churn is one of the biggest challenges in the banking industry. Retaining an existing customer is often more cost-effective than acquiring a new one.

This project predicts whether a customer is likely to churn based on customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Account Balance
* Number of Products
* Credit Card Ownership
* Active Membership
* Estimated Salary

The application provides the probability of customer churn in real time through a user-friendly Streamlit interface. The app loads a trained ANN model along with the required preprocessing objects and generates predictions instantly.

---

## ✨ Features

* Interactive Streamlit web application
* Real-time customer churn prediction
* Artificial Neural Network (ANN) built using TensorFlow/Keras
* Label Encoding and One-Hot Encoding
* Feature Standardization using StandardScaler
* Displays churn probability
* Clean and responsive user interface
* Publicly deployed application

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* Matplotlib
* Pickle

---

## 🧠 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Label Encoding
6. One-Hot Encoding
7. Feature Scaling
8. ANN Model Training
9. Model Evaluation
10. Model Serialization
11. Streamlit Deployment

---

## 📂 Dataset

**Dataset:** Bank Customer Churn Modelling Dataset

**Target Variable:**

* Exited

  * **1** → Customer Churned
  * **0** → Customer Stayed

---

## 📷 Application Preview

### Home Screen

> *(Add your Streamlit homepage screenshot here.)*

### Prediction Result

> *(Add your prediction result screenshot here.)*

---

## ⚙️ Project Structure

```text
Customer-Churn-Prediction-ANN/
│
├── app.py
├── model.h5
├── scaler.pkl
├── onehot_encoder_geo.pkl
├── label_encoder_gender.pkl
├── Churn_Modelling.csv
├── prediction.ipynb
├── experiments.ipynb
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/niraj-codes01/customer-churn-prediction-ann.git
```

Navigate to the project:

```bash
cd customer-churn-prediction-ann
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

* User authentication
* Batch prediction using CSV upload
* Model explainability with SHAP
* Improved UI/UX
* Docker containerization
* CI/CD pipeline
* Cloud deployment with monitoring

---

## 💡 What I Learned

This project taught me much more than building an ANN.

While the model training was straightforward, deployment presented unexpected challenges. Streamlit Community Cloud initially attempted to deploy the application using a Python version that wasn't compatible with TensorFlow, resulting in repeated installation failures.

Debugging deployment logs, understanding dependency compatibility, and resolving environment issues were valuable learning experiences. It reinforced an important lesson:

> **Sometimes the problem isn't in your code—it's in the deployment environment. Reading logs carefully is just as important as writing good code.**

---

## 🤝 Feedback

If you try the application, I'd love to hear your feedback or suggestions for improvement.

If you found this project helpful, consider giving the repository a ⭐.

---

## 👨‍💻 Author

**Niraj Pachpande**

* GitHub: https://github.com/niraj-codes01
* LinkedIn: *www.linkedin.com/in/niraj-codes01*

---

### ⭐ If you found this project useful, please consider starring the repository!
