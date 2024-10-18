from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import networkx as nx
from flask_cors import CORS

# Load the pre-trained models and scaler
rf_model = joblib.load('rf_model.pkl')
nn_model = load_model('nn_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define route for diabetes prediction using ensemble model
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract features from input data
    # Convert 'Gender' from dropdown value: 0 for Male, 1 for Female
    features = np.array([
        data['Gender'],  # Gender: 0 for Male, 1 for Female
        data['Age'],
        data['BMI'],
        data['BloodPressure'],
        data['Glucose']
    ]).reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Get predictions from both models
    rf_pred = rf_model.predict(features_scaled)
    nn_pred = nn_model.predict(features_scaled)
    nn_pred = (nn_pred > 0.5).astype(int).flatten()

    # Ensemble: Majority voting
    final_pred = np.round((rf_pred + nn_pred) / 2)[0]

    # Query the knowledge graph for recommendations
    recommendation = get_recommendation(final_pred)

    # Return prediction and recommendation as JSON
    return jsonify({
        'prediction': 'Diabetic' if final_pred == 1 else 'Not Diabetic',
        'recommendation': recommendation
    })

# Function to query the knowledge graph
def get_recommendation(prediction):
    G = nx.Graph()
    # Example knowledge graph for diabetes management recommendations
    G.add_edge('Diabetic', 'Regular Exercise', recommendation='30 mins daily')
    G.add_edge('Diabetic', 'Diet Control', recommendation='Low sugar, balanced diet')
    G.add_edge('Not Diabetic', 'Maintain Weight', recommendation='Regular weight checks')

    if prediction == 1:
        return {
            'exercise': G['Diabetic']['Regular Exercise']['recommendation'],
            'diet': G['Diabetic']['Diet Control']['recommendation']
        }
    else:
        return {'recommendation': G['Not Diabetic']['Maintain Weight']['recommendation']}

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
