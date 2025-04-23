# 🧪 First ML Model – Solubility Prediction

Yeh project ek **Machine Learning based Regression Model** hai jo chemical compounds ki **solubility** (logS) predict karta hai, using the **Delaney Dataset**.

## 🎯 Project Ka Aim – Kya Banana Hai?

Is project ka goal hai ek aisa ML model banana jo kisi bhi chemical compound ke features (jaise molecular weight, number of bonds etc.) ke basis par bata sake ki uski solubility kitni hogi. Yeh feature **drug discovery** aur **chemical analysis** jaisi fields mein kaafi kaam aata hai.

---

## 🔍 Step-by-Step Explanation (Simple Hinglish Mein)

### 📦 1. Data Load aur Preprocessing

```python
data = pd.read_csv("delaney.csv")
```

- `pandas` se CSV file load ki gayi.

```python
data.columns
```

- Columns (features) check kiye.

```python
X = data.drop("logS", axis=1)
y = data["logS"]
```

- `X` mein features aur `y` mein target variable (logS).

---

### 🔀 2. Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

- 80% training data, 20% testing ke liye.

---

### 🤖 3. Model Training

#### Linear Regression

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
```

- Simple linear model jo straight line fit karta hai.

#### Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
```

- Ensemble model jo multiple decision trees use karta hai.

---

### 📊 4. Model Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score
```

- Performance measure karne ke liye metrics.

```python
y_pred_lr = lr.predict(X_test)
y_pred_rf = rf.predict(X_test)
```

- Models ke predictions.

```python
print("LR MSE:", mean_squared_error(y_test, y_pred_lr))
print("RF MSE:", mean_squared_error(y_test, y_pred_rf))
```

- MSE compare kiya.

```python
print("LR R2:", r2_score(y_test, y_pred_lr))
print("RF R2:", r2_score(y_test, y_pred_rf))
```

- R2 score compare kiya.

---

### 📈 5. Visualization – Actual vs Predicted

```python
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel("Actual Solubility (logS)")
plt.ylabel("Predicted Solubility (logS)")
plt.title("Random Forest Predictions")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.show()
```

- Graph se dikhaya ki predictions aur real values kitne close hain.

---

## 🔧 Required Libraries

```bash
pip install pandas scikit-learn matplotlib
```

---

## 🏃‍♂️ Kaise Chalayein?

1. Is repo ko clone karo ya `.ipynb` file ko Google Colab mein open karo.
2. Required libraries install karo.
3. Notebook step-by-step run karo.
4. Graph aur outputs ko samjho.

---

## 🧠 Future Ideas

- Advanced models try karo jaise XGBoost, SVR.
- Hyperparameter tuning karo.
- Feature selection techniques add karo.

---
