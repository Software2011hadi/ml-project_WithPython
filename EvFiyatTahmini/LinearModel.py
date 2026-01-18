import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("evler_200.csv")

X = df[["m2","oda","yas","uzaklik","kira"]]
y = df["fiyat"]

model = LinearRegression()
model.fit(X,y)

x_ = [[152,4,18,14.06,1467]]

pred=model.predict(x_)
print(pred)






