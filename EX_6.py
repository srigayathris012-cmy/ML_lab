import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
data = {
    "Age": [22, 25, 30, 35, 40, 28],
    "Browsing History": [2, 3, 5, 6, 7, 4],
    "Time Spent": [5, 8, 15, 20, 25, 10],
    "Click": [0, 0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)
print("------ DATASET ------")
print(df.to_string(index=False))
X = df[["Age", "Browsing History", "Time Spent"]]
y = df["Click"]
model = GaussianNB()
model.fit(X, y)
age = int(input("\nEnter Age: "))
history = int(input("Enter Browsing History: "))
time = int(input("Enter Time Spent on Website: "))
new_user = pd.DataFrame(
    [[age, history, time]],
    columns=["Age", "Browsing History", "Time Spent"]
)
print("\nPredicted Click:", prediction[0])
if prediction[0] == 1:
    print("User is likely to CLICK the advertisement")
else:
    print("User is NOT likely to CLICK the advertisement")
probability = model.predict_proba(new_user)
print("\nProbability of No Click:", probability[0][0])
print("Probability of Click:", probability[0][1])
features = ["Age", "Browsing History", "Time Spent"]
feature_importance = abs(model.theta_[0] - model.theta_[1])
plt.figure(figsize=(6, 6))
plt.bar(features, feature_importance)
plt.title("Feature Importance - Naive Bayes")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.grid(axis="y")
plt.show()
