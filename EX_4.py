import pandas as pd
import matplotlib.pyplot as plt
from math import log2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    "StudyHours": [2, 3, 4, 5, 6, 7, 8, 1, 9, 10, 5, 6, 4, 8, 3],
    "Attendance": [55, 60, 65, 70, 75, 80, 85, 50, 90, 95, 72, 78, 68, 88, 58],
    "Result": [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

pass_count = (df["Result"] == 1).sum()
fail_count = (df["Result"] == 0).sum()
total = len(df)

p_pass = pass_count / total
p_fail = fail_count / total

entropy = -(p_pass * log2(p_pass)) - (p_fail * log2(p_fail))

print("----------- Dataset -----------")
print("Total Students :", total)
print("Pass Students  :", pass_count)
print("Fail Students  :", fail_count)
print("Entropy        :", round(entropy, 3))


X = df[["StudyHours", "Attendance"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy :", round(accuracy * 100, 2), "%")

print("\n------ Student Result Prediction ------")

study = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance Percentage: "))

new_student = pd.DataFrame({
    "StudyHours": [study],
    "Attendance": [attendance]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction : PASS")
else:
    print("Prediction : FAIL")

plt.figure(figsize=(15, 8))

plot_tree(
    model,
    feature_names=["StudyHours", "Attendance"],
    class_names=["Fail", "Pass"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree using Entropy (ID3)")
plt.show()
