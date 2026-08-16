import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Dataset

data = {
    "Age": [25, 28, 30, 32, 35, 37, 40, 42, 45, 48,
            50, 52, 55, 58, 60, 62, 65, 68, 70, 72],

    "BMI": [20.5, 21.8, 22.5, 23.2, 24.0, 25.1, 26.5, 27.3, 28.4, 29.2,
            30.0, 30.8, 31.5, 32.2, 33.0, 33.8, 34.5, 35.0, 35.8, 36.5],

    "BloodPressure": [110, 112, 115, 118, 120, 122, 125, 128, 130, 132,
                      135, 138, 140, 142, 145, 148, 150, 152, 155, 158],

    "Cholesterol": [160, 165, 170, 175, 180, 185, 190, 195, 205, 215,
                    225, 230, 235, 240, 245, 250, 255, 260, 265, 270],

    "Diabetes": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[["Age", "BMI", "BloodPressure", "Cholesterol"]]
y = df["Diabetes"]

# Train Model
model = LogisticRegression()
model.fit(X, y)

# User Input
age = float(input("Enter Age: "))
bmi = float(input("Enter BMI: "))
bp = float(input("Enter Blood Pressure: "))
chol = float(input("Enter Cholesterol: "))

# Prediction DataFrame (removes warning)
new_data = pd.DataFrame({
    "Age": [age],
    "BMI": [bmi],
    "BloodPressure": [bp],
    "Cholesterol": [chol]
})

# Prediction
prediction = model.predict(new_data)
probability = model.predict_proba(new_data)[0][1]

if prediction[0] == 1:
    print("Diabetes Detected")
else:
    print("No Diabetes")

print("Probability:", round(probability, 2))

# ---------------- GRAPH ----------------

# Average values of remaining features
avg_bmi = df["BMI"].mean()
avg_bp = df["BloodPressure"].mean()
avg_chol = df["Cholesterol"].mean()

# X-axis values (Age)
x_values = np.linspace(df["Age"].min(), df["Age"].max(), 200)

# Data for sigmoid curve
curve_data = pd.DataFrame({
    "Age": x_values,
    "BMI": avg_bmi,
    "BloodPressure": avg_bp,
    "Cholesterol": avg_chol
})

# Predicted probabilities
y_prob = model.predict_proba(curve_data)[:, 1]

# Plot
plt.figure(figsize=(6,4))

# Blue dots (Actual Data)
plt.scatter(df["Age"], y,
            color="blue",
            s=80,
            label="Actual Data")

# Red sigmoid curve
plt.plot(x_values,
         y_prob,
         color="red",
         linewidth=2,
         label="Logistic Regression Curve")

# Green star (User prediction)
plt.scatter(age,
            probability,
            color="green",
            marker="*",
            s=180,
            label="Your Prediction")

plt.xlabel("Age")
plt.ylabel("Probability")
plt.title("Logistic Regression")
plt.legend()
plt.grid(True)

plt.show()
