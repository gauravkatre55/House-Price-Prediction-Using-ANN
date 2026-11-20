# 🏠 House Price Prediction Using Artificial Neural Network (ANN)

### Deep Learning | Regression | TensorFlow | Keras | Python

## 📌 Project Overview

This project focuses on predicting house prices using an Artificial Neural Network (ANN) regression model.

The model is developed using TensorFlow and Keras to learn complex relationships between different housing features and their corresponding median house values.

The project includes data preprocessing, exploratory data analysis, feature scaling, ANN model development, model training, evaluation, and house price prediction.

## 🖥️ Application Preview

The project includes an interactive Streamlit web application
that predicts house prices using a trained Artificial Neural Network.

### House Price Prediction Application

![House Price Prediction Application](Screenshot/house-price-prediction-app.png)

## 🎯 Project Objectives

- Develop an ANN-based regression model for house price prediction.
- Understand relationships between housing features and property values.
- Perform data preprocessing and feature scaling.
- Build and train a deep learning model using TensorFlow and Keras.
- Evaluate model performance using regression metrics.
- Predict house prices based on user-provided property information.

## 📂 Dataset Information

**Dataset:** California Housing Dataset

The dataset contains housing information from California.

### Input Features

| Feature | Description |
|---|---|
| MedInc | Median income in the area |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population of the area |
| AveOccup | Average household occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

**Target Variable:** `MedHouseVal`

## 🛠️ Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- Streamlit
- Joblib / Pickle

## 🧠 ANN Model Architecture

The Artificial Neural Network consists of the following layers:

```text
       Input Layer
      (8 Features)
           |
           ▼
    Dense Layer (64)
      Activation: ReLU
           |
           ▼
    Dense Layer (32)
      Activation: ReLU
           |
           ▼
     Output Layer (1)
      Activation: Linear
           |
           ▼
    Predicted House Price
```

### Model Configuration

| Parameter | Configuration |
|---|---|
| Model Type | ANN Regression |
| Input Features | 8 |
| Hidden Layers | 2 |
| Hidden Neurons | 64, 32 |
| Hidden Activation | ReLU |
| Output Neurons | 1 |
| Output Activation | Linear |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Mean Squared Error (MSE) |
| Evaluation Metric | Mean Absolute Error (MAE) |
| Epochs | 100 |
| Batch Size | 32 |
| Early Stopping | Patience 10 |
| Feature Scaling | StandardScaler |

## ⚙️ Project Workflow

1. Dataset Collection
2. Data Loading
3. Exploratory Data Analysis (EDA)
4. Data Cleaning and Preprocessing
5. Feature Selection
6. Train-Test Splitting
7. Feature Scaling using StandardScaler
8. ANN Model Building
9. Model Compilation
10. Model Training
11. Model Evaluation
12. House Price Prediction

## 📊 Model Evaluation

The model performance is evaluated using the following regression metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

*Actual evaluation results can be added here after model training.*

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gauravkatre55/House-Price-Prediction-Using-ANN.git
```

### 2. Navigate to Project Folder

```bash
cd House-Price-Prediction-Using-ANN
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

## 💡 Key Learning Outcomes

- Understanding Artificial Neural Networks for regression problems.
- Implementing deep learning models using TensorFlow and Keras.
- Understanding forward propagation and model training.
- Applying feature scaling to numerical data.
- Evaluating regression model performance.
- Deploying a machine learning model using Streamlit.

## 🔮 Future Improvements

- Experiment with different ANN architectures.
- Perform hyperparameter tuning.
- Compare ANN performance with traditional regression algorithms.
- Improve prediction accuracy.
- Deploy the application using Streamlit Community Cloud.

## 👨‍💻 Author

**Gaurav Katre**

- GitHub: [gauravkatre55](https://github.com/gauravkatre55)
- LinkedIn: [Gaurav Katre](https://www.linkedin.com/in/gauravkatre/)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Developed by Gaurav Katre | Deep Learning Project**
