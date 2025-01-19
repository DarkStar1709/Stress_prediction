import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

# Load dataset
df = pd.read_csv('Stress_Level_Prediction_Dataset_Updated.csv')

from scipy.stats import zscore
def cap_floor_outliers(df, columns, threshold=3):
    for col in columns:
        # Calculate Z-scores
        z_scores = zscore(df[col])
        
        # Cap and floor based on the threshold
        df[col] = np.where(z_scores > threshold, 
                           df[col].mean() + threshold * df[col].std(),  # Cap high outliers
                           np.where(z_scores < -threshold, 
                                    df[col].mean() - threshold * df[col].std(),  # Floor low outliers
                                    df[col]))  # Keep non-outliers as is
    return df

# Apply to the specified columns
columns_to_process = ['Heart_Rate', 'Diastolic_BP', 'Systolic_BP', 'Pulse_Rate']
threshold = 3  # Define the Z-score threshold
df = cap_floor_outliers(df, columns_to_process, threshold)

correlation_matrix = df.corr(method='pearson')  # Use 'spearman' or 'kendall' for non-linear relationships
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

threshold = 0.85
upper_triangle = correlation_matrix.where(
    ~np.tril(np.ones(correlation_matrix.shape, dtype=bool))
)

# Find features to drop
features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
df = df.drop(columns=features_to_drop)

print("Dropped features:", features_to_drop)

X = df.drop(columns=["Stress_Level"])  # Input features
y = df["Stress_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_dim=X_train.shape[1]),  # First hidden layer
    Dense(8, activation='relu'),                             # Second hidden layer
    Dense(4, activation='relu'),
    Dense(1, activation='linear')                             # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
Ann_model = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Predict on new data
y_pred = model.predict(X_test)

#save the model
pickle.dump(model, open('model.pkl', 'wb'))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2}")

import numpy as np

# Define the number of features (based on your dataset)
num_features = X_test.shape[1]

# Take input from the user
user_input = input(f"Enter {num_features} values separated by commas: ")

# Process the input
user_data = np.array([float(x) for x in user_input.split(',')]).reshape(1, -1)

# Check if the input matches the expected feature size
if user_data.shape[1] != num_features:
    print(f"Error: Expected {num_features} features, but got {user_data.shape[1]}.")
else:
    # Make prediction
    prediction = model.predict(user_data)
    if prediction > 20 :
            prediction = 19.2
    print(f"Prediction: {prediction}")