import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ----------------------------
# Load the trained model
# ----------------------------
trained_model = load_model("dqn_diabetes_model.h5", compile=False)
trained_model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')

# ----------------------------
# Load and fit the scaler on training data
# ----------------------------
# In production, you should load a pre-fitted scaler, e.g., using joblib.load("scaler.save")
data = pd.read_csv("synthetic_ehr_data.csv")
scaler = StandardScaler()
scaler.fit(data[['Age', 'Glucose', 'HbA1c', 'Systolic_BP', 'Diastolic_BP', 'BMI', 'Exercise']])

# ----------------------------
# Define the actions dictionary
# ----------------------------
actions = {
    0: "Maintain Current Medication & Lifestyle",
    1: "Increase Insulin Dosage",
    2: "Decrease Insulin Dosage",
    3: "Switch to Alternative Medication",
    4: "Suggest Lifestyle Changes (Diet, Exercise)",
    5: "Immediate Doctor Consultation Required"
}

# ----------------------------
# /predict endpoint: Receives patient EHR data and returns a treatment recommendation
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json()

    # Expected JSON payload keys: Age, Glucose, HbA1c, Systolic_BP, Diastolic_BP, BMI, Medication, Exercise
    # Build a patient data list in the same order as training:
    patient_data = [
        data_json.get("Age"),
        data_json.get("Glucose"),
        data_json.get("HbA1c"),
        data_json.get("Systolic_BP"),
        data_json.get("Diastolic_BP"),
        data_json.get("BMI"),
        data_json.get("Medication"),  # This is used as raw (not scaled)
        data_json.get("Exercise")
    ]

    # Scale the features that were scaled during training:
    # We assume columns: Age, Glucose, HbA1c, Systolic_BP, Diastolic_BP, BMI, and Exercise need scaling,
    # while Medication remains raw.
    # Create an array for the features to scale:
    features_to_scale = np.array([[ 
        patient_data[0],  # Age
        patient_data[1],  # Glucose
        patient_data[2],  # HbA1c
        patient_data[3],  # Systolic_BP
        patient_data[4],  # Diastolic_BP
        patient_data[5],  # BMI
        patient_data[7]   # Exercise
    ]])
    
    scaled_features = scaler.transform(features_to_scale)
    
    # Reconstruct final state vector in training order:
    # Order: [Age_scaled, Glucose_scaled, HbA1c_scaled, Systolic_BP_scaled, Diastolic_BP_scaled, BMI_scaled, Medication (raw), Exercise_scaled]
    final_state = np.hstack((scaled_features[0][:6], [patient_data[6]], [scaled_features[0][6]]))
    final_state = final_state.reshape(1, -1)

    # Get Q-values from the model and select the best action:
    q_values = trained_model.predict(final_state, verbose=0)
    action_id = int(np.argmax(q_values[0]))
    recommended_action = actions.get(action_id, "No Recommendation Available")

    return jsonify({"recommendation": recommended_action})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
