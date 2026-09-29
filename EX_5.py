from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
X = [
    [45, 120, 180],  
    [50, 130, 200],  
    [55, 140, 220],  
    [60, 150, 240],  
    [48, 125, 190],  
    [58, 145, 230]   
]
y = [
    "No",    
    "No",    
    "Yes",   
    "Yes",  
    "No",    
    "Yes"    
]
df = pd.DataFrame(X, columns=["Age", "Blood Pressure", "Cholesterol"])
df["Heart Disease"] = y
print("\n----- DATASET -----")
print(df)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
age = float(input("Enter Age: "))
bp = float(input("Enter Blood Pressure: "))
cholesterol = float(input("Enter Cholesterol Level: "))
new_patient = [[age, bp, cholesterol]]
prediction = knn.predict(new_patient)
print("\nPredicted Heart Disease:", prediction[0])
yes_patients = [X[i] for i in range(len(X)) if y[i] == "Yes"]
no_patients = [X[i] for i in range(len(X)) if y[i] == "No"]
plt.scatter(
    [p[0] for p in yes_patients],
    [p[2] for p in yes_patients],
    label="Heart Disease: Yes"
)
plt.scatter(
    [p[0] for p in no_patients],
    [p[2] for p in no_patients],
    label="Heart Disease: No"
)
plt.scatter(
    age,
    cholesterol,
    marker="*",
    s=200,
    label="New Patient"
)
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.title("KNN Heart Disease Classification")
plt.legend()
plt.grid(True)
plt.show()
