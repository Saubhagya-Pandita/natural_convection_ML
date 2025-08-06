import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

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

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the final model with the best parameters
svr_best = SVR(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
svr_best.fit(X_train, y_train)

# Validate the final model
y_val_pred = svr_best.predict(X_val)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

# Test the final model
y_test_pred = svr_best.predict(X_test)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Plotting in 3D for validation set
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for validation data
ax.scatter(X_val[:, 0], X_val[:, 1], y_val, color='blue', label='Actual', marker='o')

# Predictions on validation data
val_pred = svr_best.predict(X_val)
ax.scatter(X_val[:, 0], X_val[:, 1], val_pred, color='red', label='Predicted', marker='^')

# Plot hyperplane
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
z_grid = svr_best.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
z_grid = z_grid.reshape(x_grid.shape)
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap='viridis')

# Labels and title
ax.set_xlabel('Orientation')
ax.set_ylabel('Voltage')
ax.set_zlabel('Nusselt')
ax.set_title('SVR: Actual vs Predicted (Validation)')
ax.legend()

plt.show()

# Plotting in 3D for test set
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for test data
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='green', label='Actual', marker='o')

# Predictions on test data
test_pred = svr_best.predict(X_test)
ax.scatter(X_test[:, 0], X_test[:, 1], test_pred, color='purple', label='Predicted', marker='^')

# Plot hyperplane
ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap='viridis')

# Labels and title
ax.set_xlabel('Orientation')
ax.set_ylabel('Voltage')
ax.set_zlabel('Nusselt')
ax.set_title('SVR: Actual vs Predicted (Test)')
ax.legend()

plt.show()

# Print results
print("Validation Results:")
print(f"MAE: {val_mae}")
print(f"RMSE: {val_rmse}")
print(f"R²: {val_r2}")

print("\nTest Results:")
print(f"MAE: {test_mae}")
print(f"RMSE: {test_rmse}")
print(f"R²: {test_r2}")
