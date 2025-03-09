import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Engine Performance Prediction: Horsepower vs. Torque
data_engine = {
    'Torque (Nm)': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    'Horsepower (HP)': [75, 110, 150, 185, 220, 260, 300, 340, 375, 410]
}
df_engine = pd.DataFrame(data_engine)
X = df_engine[['Torque (Nm)']]
y = df_engine['Horsepower (HP)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_engine = LinearRegression()
model_engine.fit(X_train, y_train)
y_pred = model_engine.predict(X_test)

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Regression Line')
plt.xlabel('Torque (Nm)')
plt.ylabel('Horsepower (HP)')
plt.title('Linear Regression - Engine Performance')
plt.legend()
plt.grid()
plt.show()

print("Engine Performance Prediction")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))
