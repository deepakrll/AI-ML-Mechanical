import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 2. Material Strength Analysis: Stress vs. Strain
data_material = {
    'Strain (%)': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'Stress (MPa)': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
}
df_material = pd.DataFrame(data_material)
X = df_material[['Strain (%)']]
y = df_material['Stress (MPa)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_material = LinearRegression()
model_material.fit(X_train, y_train)
y_pred = model_material.predict(X_test)

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_test, y_pred, color='green', linewidth=2, label='Regression Line')
plt.xlabel('Strain (%)')
plt.ylabel('Stress (MPa)')
plt.title('Linear Regression - Material Strength Analysis')
plt.legend()
plt.grid()
plt.show()

print("Material Strength Analysis")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))
