import pandas as pd
data = pd.read_csv("breast-cancer.csv")


data.head()
data.shape
data.info()
data.describe()


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(data.isnull())
plt.show()
data.isnull().sum()


data.columns()
data = data.drop(["id"], axis=1)


data["diagnosis"].unique()
data["diagnosis"] = data["diagnosis"].replace({"M" : 1, "B" : 0})
data["diagnosis"] = data["diagnosis"].astype("float64")
data["diagnosis"].count_values()


from sklearn.model_selection import train_test_split
x = data.drop(["diagnosis"], axis=1)
y = data["diagnosis"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)


test = model.predict(x_test)


from sklearn.metrics import accuracy_score
result = accuracy_score(y_test, test)
print(result)


import pickle
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)