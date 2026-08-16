import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Bedrooms": [2, 3, 4, 5, 3],
    "Size": [1000, 1500, 1800, 2500, 1600],
    "Age": [10, 5, 8, 2, 6],
    "Price": [30, 45, 55, 75, 50]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[["Bedrooms", "Size", "Age"]]
y = df["Price"]

# Train Model
model = LinearRegression()
model.fit(X, y)

# User Input
bedrooms = int(input("Enter bedrooms: "))
size = int(input("Enter house size: "))
age = int(input("Enter house age: "))

# Prediction
new_house = pd.DataFrame({
    "Bedrooms": [bedrooms],
    "Size": [size],
    "Age": [age]
})

pred = model.predict(new_house)

print("Predicted Price:", pred[0])

# Plot
plt.scatter(y, model.predict(X), color="blue")
plt.plot(y, y, color="red")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression")

plt.show()
