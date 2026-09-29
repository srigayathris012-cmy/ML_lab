import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
data = {
    "Email Length": [120, 150, 100, 60, 80],
    "Number of Links": [5, 6, 4, 0, 1],
    "Spam": [1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)
print("----- DATASET -----")
print(df.to_string(index=False))
X = df[["Email Length", "Number of Links"]]
y = df["Spam"]
model = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
model.fit(X, y)
length = float(input("\nEnter Email Length: "))
links = float(input("Enter Number of Links: "))

new_email = pd.DataFrame(
    [[length, links]],
    columns=["Email Length", "Number of Links"]
)

prediction = model.predict(new_email)

print("\nPredicted Class:", prediction[0])

if prediction[0] == 1:
    print("Email is SPAM")
else:
    print("Email is NOT SPAM")
features = ["Email Length", "Number of Links"]
importance = abs(model.coef_[0])

plt.figure(figsize=(8, 6))

plt.scatter(
    df[df["Spam"] == 0]["Email Length"],
    df[df["Spam"] == 0]["Number of Links"],
    label="Not Spam"
)

plt.scatter(
    df[df["Spam"] == 1]["Email Length"],
    df[df["Spam"] == 1]["Number of Links"],
    label="Spam"
)
plt.scatter(
    length,
    links,
    marker="*",
    s=200,
    label="New Email"
)

plt.xlabel("Email Length")
plt.ylabel("Number of Links")
plt.title("Perceptron Spam Classification")
plt.legend()
plt.grid()
plt.show()
