import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\ML_Projects\EvFiyatTahmini\evler_200.csv")

X = df[["m2","oda","yas","uzaklik","kira"]]
y = df["fiyat"]
X_train , X_test , y_train , y_test = train_test_split(
    X,y,test_size=0.2,random_state=2
)
model = LinearRegression()
model.fit(X_train,y_train)

train_score = model.score(X_train,y_train)
test_score = model.score(X_test,y_test)

x_ = [[182,4,18,14.06,1467]]

pred=model.predict(x_)
print("Tahmin: ",pred)
print(f"Test Score: {test_score:.3f}")
print(f"Train Score: {train_score:.3f}")





