from flask import Flask, request, jsonify
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the pickled model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None  # or exit or handle in some way

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # Parse input JSON
        data = request.json
        input_features = data.get("features")
        
        if not input_features:
            return jsonify({"error": "No input features provided"}), 400

        # Convert input to numpy array
        user_data = np.array(input_features).reshape(1, -1)

        # Validate input size
        if user_data.shape[1] != model.input_shape[1]:
            return jsonify({"error": f"Expected {model.input_shape[1]} features, but got {user_data.shape[1]}."}), 400

        # Make prediction
        prediction = model.predict(user_data)
        prediction = prediction.flatten()[0]  # Extract single prediction value

        # Apply post-processing if needed
        if prediction > 20:
            prediction = 19.2

        # Convert to Python native type
        prediction = int(prediction)

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define a test route
@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Stress Prediction API!"

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
