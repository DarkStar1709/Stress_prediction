import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Load dataset
df = pd.read_csv('Stress_Level_Prediction_Dataset_Updated.csv')

# Function to cap and floor outliers
def cap_floor_outliers(df, columns, threshold=3):
    for col in columns:
        z_scores = zscore(df[col])
        df[col] = np.where(
            z_scores > threshold,
            df[col].mean() + threshold * df[col].std(),  # Cap high outliers
            np.where(
                z_scores < -threshold,
                df[col].mean() - threshold * df[col].std(),  # Floor low outliers
                df[col]  # Keep non-outliers as is
            )
        )
    return df

# Apply outlier processing
columns_to_process = ['Heart_Rate', 'Diastolic_BP', 'Systolic_BP', 'Pulse_Rate']
threshold = 3
df = cap_floor_outliers(df, columns_to_process, threshold)

# Correlation analysis
correlation_matrix = df.corr(method='pearson')
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Drop highly correlated features
threshold = 0.85
upper_triangle = correlation_matrix.where(~np.tril(np.ones(correlation_matrix.shape, dtype=bool)))
features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
df = df.drop(columns=features_to_drop)
print("Dropped features:", features_to_drop)

# Splitting data
X = df.drop(columns=["Stress_Level"])  # Input features
y = df["Stress_Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the TensorFlow model
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train.shape[1]),  # First hidden layer
    Dense(8, activation='relu'),                              # Second hidden layer
    Dense(4, activation='relu'),                              # Third hidden layer
    Dense(1, activation='linear')                             # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
Ann_model = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# Predict on test data
y_pred = model.predict(X_test)

# Save the TensorFlow model using `model.save`
model.save('model.h5')  # Save the model in HDF5 format

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

# Test with user input
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
    # Load the model and make predictions
    model = load_model('model.h5')  # Load the saved TensorFlow model
    prediction = model.predict(user_data)
    if prediction > 20:
        prediction = 19.2
    print(f"Prediction: {prediction}")