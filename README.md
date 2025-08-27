# Car Price Prediction (Linear Regression from Scratch)

This project implements **multivariate linear regression** from scratch using **NumPy**.  
The goal is to predict car prices based on features like mileage ,age ,horsepower ,wheelbase and price. without using machine learning libraries like `scikit-learn`.

---

## 📌 Project Overview
- Load and preprocess training data  
- Apply **z-score normalization** to scale features  
- Implement **Mean Squared Error cost function**  
- Implement **gradient descent optimization**  
- Train a regression model to learn weights and bias  
- Make predictions for new inputs  
- Visualize results with plots  

This is a learning project to understand the internals of regression models.

---

## 📂 Repository Structure
<pre>Car_price_prediction/
│
├── data/
│ └── Training Data.txt # Dataset (car features + price)
│
├── car_price_prediction.py # Main Python script
├── results/
│ ├── features_vs_price.png # Scatter plots of features vs price
│ ├── cost_convergence.png # Cost function over iterations
│ ├── actual_vs_pred.png # Actual vs Predicted prices
│
└── README.md # Project description
  </pre>

---

## ⚙️ Features Implemented
- Data loading & visualization  
- Z-score normalization  
- Cost function (MSE)  
- Gradient calculation  
- Gradient descent optimization  
- Prediction function  
- Evaluation plots  

---

## 📊 Example Plots
### 1. Features vs Price
Shows how each feature (mileage ,age ,horsepower ,wheelbase) affect on car price.

### 2. Cost Function Convergence
Cost decreases over iterations, showing gradient descent optimization.

### 3. Actual vs Predicted Prices
Comparison of model predictions with actual car prices.

---

## 🚀 How to Run
1. Clone the repository:
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
2. Install requirements (only NumPy & Matplotlib needed):
   pip install numpy matplotlib 
3. Run the script:
   python car_price_prediction.py

---
   
## 📈 Sample Prediction
1. Example input:
   inp = np.array([30000, 3, 130, 88.6])  # mileage, age, horsepower, size
2. Output:
   Predicted price ≈ 21500
