import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from hpelm import ELM

# Load the dataset
data = pd.read_csv('new_data_points.csv')

# Splitting the dataset into features (X) and target (y)
X = data[['Orientation', 'Voltage']].values
y = data['Nusselt'].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (70%), validation (15%), and testing (15%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Hyperparameter tuning
best_neurons = 0
best_val_mae = float('inf')

# Reduced maximum number of neurons to 50
for neurons in range(5, 51, 5):
    elm = ELM(X.shape[1], 1)
    elm.add_neurons(neurons, "sigm")
    elm.train(X_train, y_train, "r")
    
    # Validate the model
    y_val_pred = elm.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_neurons = neurons

print(f"Best number of neurons: {best_neurons}")

# Train the final model with the best number of neurons
elm = ELM(X.shape[1], 1)
elm.add_neurons(best_neurons, "sigm")
elm.train(X_train, y_train, "r")

# Validate the final model
y_val_pred = elm.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

# Test the final model
y_test_pred = elm.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print("Validation Results:")
print(f"MAE: {val_mae}")
print(f"RMSE: {val_rmse}")
print(f"R²: {val_r2}")

print("\nTest Results:")
print(f"MAE: {test_mae}")
print(f"RMSE: {test_rmse}")
print(f"R²: {test_r2}")
