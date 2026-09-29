import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Dataset
data = {
    "Age": [25, 30, 35, 45, 50],
    "Tumor Size": [2, 3, 4, 7, 8],
    "Diagnosis": [0, 0, 0, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

print("----- DATASET -----")
print(df.to_string(index=False))

# Features
X = df[["Age", "Tumor Size"]]

# Target
y = df["Diagnosis"]

# Create SVM model
model = SVC(kernel="linear")

# Train model
model.fit(X, y)

# User input
age = float(input("\nEnter Age: "))
size = float(input("Enter Tumor Size: "))

# New patient
new_patient = [[age, size]]

# Prediction
prediction = model.predict(new_patient)

print("\nPredicted Class:", prediction[0])

if prediction[0] == 1:
    print("Diagnosis: Malignant")
else:
    print("Diagnosis: Benign")

# Graph
plt.figure(figsize=(8, 6))

plt.scatter(
    df[df["Diagnosis"] == 0]["Age"],
    df[df["Diagnosis"] == 0]["Tumor Size"],
    label="Benign"
)

plt.scatter(
    df[df["Diagnosis"] == 1]["Age"],
    df[df["Diagnosis"] == 1]["Tumor Size"],
    label="Malignant"
)

plt.scatter(age, size, marker="*", s=200, label="New Patient")

plt.xlabel("Age")
plt.ylabel("Tumor Size")
plt.title("SVM Classification")
plt.legend()
plt.grid()

plt.show()
