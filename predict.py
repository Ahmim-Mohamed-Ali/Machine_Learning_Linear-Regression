import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/advertising.csv")

# On Ajoute Un Terme Quadratique pour notre variable tv 
df['tv2'] = df.tv**2


scaler = MinMaxScaler()
scaler.fit(df)
data_array = scaler.transform(df)
df = pd.DataFrame(data_array, columns = ['tv','radio','journaux','ventes','tv2'])

# Création Du Modele 
reg = LinearRegression()
X = df[['tv','radio','journaux','tv2']]
y = df.ventes


# Creations des Echantillon De Test Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Entrainement Du Modèle
reg.fit(X_train, y_train)


# Prediction Du Modele 
y_pred_test = reg.predict(X_test)


# Affichage Des Scores D'évaluations Du Modele 
print(f"Coefficients: {reg.coef_}")
print(f"RMSE: {mean_squared_error(y_test, y_pred_test)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")

""" Explorons rapidement les données avec
print(df.head())
print(df.describe())
print(df.corr()) """

