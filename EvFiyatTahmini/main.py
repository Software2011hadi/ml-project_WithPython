import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(r"C:\Users\Lenovo\Desktop\ML_Projects\EvFiyatTahmini\evler_200.csv")

X = df[["m2","oda","yas","uzaklik","kira"]]
y = df["fiyat"]
X_train , X_test , y_train , y_test = train_test_split(
    X,y,test_size=0.2,random_state=2
)
model = LinearRegression()
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test) #we dont fit xtest set because it cause dataleakage.we cant see the real test score 

model.fit(x_train_scaled,y_train)

train_score = model.score(x_train_scaled,y_train)
test_score = model.score(x_test_scaled,y_test)

x_new = [[182,4,18,14.06,1467]]
x_new_scaled = scaler.transform(x_new)
pred=model.predict(x_new_scaled)
print("Tahmin: ",pred)
print(f"Test Score: {test_score:.3f}")
print(f"Train Score: {train_score:.3f}")


