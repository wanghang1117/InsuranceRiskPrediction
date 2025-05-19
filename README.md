# 🏥 Insurance Risk Prediction

This project aims to predict individual insurance charges using demographic and lifestyle features. It is structured as an end-to-end data science pipeline, from data analysis and preprocessing to model evaluation and interpretation.

---

## 📘 Project Overview

Medical insurance cost is often affected by a combination of factors such as age, smoking habits, BMI, and geographic region. In this project, we analyze a publicly available dataset and train machine learning models to predict the cost of medical insurance.

---

## 🎯 Goal

- Understand which features have the most influence on insurance costs.
- Build regression models that can predict `charges` with minimal error.
- Compare the performance of various models using evaluation metrics.

---

## 🧠 Key Learnings

- Feature engineering plays a major role in model performance.
- Categorical encoding and scaling significantly affect tree vs linear models.
- Model interpretation (e.g., feature importance) provides useful business insights.

---

## 🧰 Tools & Libraries Used

| Category         | Tools                                      |
|------------------|---------------------------------------------|
| Language         | Python 3.x                                  |
| Data Analysis    | `pandas`, `numpy`                           |
| Visualization    | `matplotlib`, `seaborn`                     |
| ML Algorithms    | `scikit-learn`, `xgboost`                   |
| Environment      | Jupyter Notebook                            |

---

## 📁 Dataset Description

Each row represents an individual with:

| Column Name | Description |
|-------------|-------------|
| `age`       | Age of primary insurance holder |
| `sex`       | Gender (`male`, `female`) |
| `bmi`       | Body Mass Index |
| `children`  | Number of dependents |
| `smoker`    | Smoking status (`yes`, `no`) |
| `region`    | Residential area in the U.S. |
| `charges`   | Total medical insurance charges (target) |

---

## 📊 Workflow

### 1. **Exploratory Data Analysis (EDA)**

- Checked for missing values and data types  
- Visualized distributions and outliers  
- Analyzed correlation between numerical features  
- Investigated how `smoker`, `bmi`, and `age` affect `charges`

### 2. **Preprocessing**

- One-hot encoding for `sex`, `smoker`, and `region`  
- Feature scaling using `StandardScaler`  
- Split data into training and testing sets (80/20 split)

### 3. **Modeling**

We experimented with the following regression models:

| Model              | Notes |
|-------------------|-------|
| Linear Regression | Baseline model |
| Decision Tree     | Captures non-linearities, prone to overfitting |
| Random Forest     | Ensemble method, better generalization |
| XGBoost           | Gradient boosting with regularization |

### 4. **Evaluation**

Used these metrics to compare models:

- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  
- **R² Score (Goodness of Fit)**

### 5. **Model Interpretation**

- Extracted feature importances from tree-based models  
- Identified top predictors: `smoker`, `bmi`, `age`

---

## 📌 Notable Insights

- **Smoker** status has the strongest impact on cost. Smokers are charged significantly higher.
- **BMI** and **Age** show a strong positive correlation with insurance charges.
- Surprisingly, **region** has minimal influence on the cost.

---

## 📈 Results

| Model          | RMSE     | R² Score |
|----------------|----------|----------|
| Linear         | 6050.12  | 0.75     |
| Random Forest  | 4123.45  | 0.86     |
| XGBoost        | **3898.22**  | **0.88**     |

> 💡 XGBoost provided the best overall performance with the lowest error.

---

## 📂 File Structure

```
insurance-prediction/
│
├── insurance_prediction.ipynb     # Main notebook
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── data/
    └── insurance.csv              # Source dataset (if available)
```

---

## 🔮 Future Work

- 📌 Use GridSearchCV or Optuna for hyperparameter tuning  
- 📌 Apply cross-validation to reduce variance in model evaluation  
- 📌 Deploy the model via Flask or Streamlit for interactive use  
- 📌 Add SHAP analysis for interpretability  

---

## 💻 How to Use

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/insurance-prediction.git
cd insurance-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook:**
```bash
jupyter notebook insurance_prediction.ipynb
```

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Hang Wang**  
📧 Email: [your_email@example.com]  
🌐 GitHub: [https://github.com/wanghang1117](https://github.com/wanghang1117)
