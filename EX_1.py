import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\sri41\OneDrive\Documents\ML_lab\temperature_sales.csv")
print("Dataset")
print(data)
X = data[["Temperature (°C)"]]
y = data["Sales (units)"]

model = LinearRegression()
model.fit(X, y)

print("\nSlope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

temp = float(input("\nEnter Temperature (°C): "))


new_data = pd.DataFrame({"Temperature (°C)": [temp]})
predicted_sales = model.predict(new_data)

# Weather prediction
if temp <= 22:
    weather = "Cloudy"
elif temp <= 38:
    weather = "Sunny"
else:
    weather = "Hot"

# Prediction result
print("\n----- Prediction Result -----")
print("Temperature:", temp, "°C")
print("Predicted Sales:", round(predicted_sales[0], 2), "units")
print("Expected Weather:", weather)

plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.xlabel("Temperature")
plt.ylabel("Sales")
plt.title("Simple Linear Regression")

plt.show()


