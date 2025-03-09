import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Simulated dataset for predictive maintenance
np.random.seed(42)
data = {
    'Temperature': np.random.randint(60, 120, 100),
    'Vibration': np.random.uniform(0.5, 5.0, 100),
    'Usage_Hours': np.random.randint(100, 5000, 100),
    'Time_To_Failure': np.random.randint(10, 1000, 100)  # Time until next failure
}
df = pd.DataFrame(data)

# Splitting data
X = df[['Temperature', 'Vibration', 'Usage_Hours']]
y = df['Time_To_Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Scatter plot with regression line
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
plt.xlabel("Actual Time To Failure")
plt.ylabel("Predicted Time To Failure")
plt.title(f"Predictive Maintenance (MAE: {mae:.2f}, MSE: {mse:.2f})")
plt.show()

# Error Distribution Plot
plt.figure(figsize=(8, 6))
sns.histplot(y_test - y_pred, bins=20, kde=True)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution in Time to Failure Prediction")
plt.show()
