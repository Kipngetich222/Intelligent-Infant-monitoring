from flask import Flask, render_template, jsonify
import random
import numpy as np
from model import build_nn_model, preprocess_data, update_model

app = Flask(__name__, static_url_path='/static')

# Define global variables
INPUT_SIZE = 4  # Temperature, Respiration, Oxygen Saturation, Movement
MODEL_UPDATE_THRESHOLD = 10  # Update the model every 10 data points

# Initialize model
model = build_nn_model()
data_counter = 0  # Counter to track the number of data points

# Simulate realistic ranges for infant vital signs
NORMAL_TEMPERATURE_RANGE = (36.5, 37.5)  # Temperature range in degrees Celsius
NORMAL_RESPIRATION_RATE_RANGE = (25, 50)  # Respiration rate range in breaths per minute
NORMAL_OXYGEN_SATURATION_RANGE = (95, 100)  # Oxygen saturation range in percentage

# Simulate abnormal ranges for alarms
ABNORMAL_TEMPERATURE_RANGE = (36.0, 38.0)
ABNORMAL_RESPIRATION_RATE_RANGE = (20, 60)
ABNORMAL_OXYGEN_SATURATION_RANGE = (90, 100)

# Simulate movements (binary: 0 for no movement, 1 for movement)
MOVEMENT = random.randint(0, 1)

@app.route('/')
def home():
    return render_template('index.html', movement=MOVEMENT)

@app.route('/get-sensor-data')
def get_sensor_data():
    global model, data_counter
    
    # Simulate sensor data within the specified ranges
    temperature = round(random.uniform(*NORMAL_TEMPERATURE_RANGE), 1)
    respiration_rate = random.randint(*NORMAL_RESPIRATION_RATE_RANGE)
    oxygen_saturation = random.randint(*NORMAL_OXYGEN_SATURATION_RANGE)
    
    # Generate abnormal sensor readings once in a while
    if random.random() < 0.7:  # 10% chance of generating abnormal data
        temperature = round(random.uniform(*ABNORMAL_TEMPERATURE_RANGE), 1)
    if random.random() < 0.3:
        respiration_rate = random.randint(*ABNORMAL_RESPIRATION_RATE_RANGE)
    if random.random() < 0.3:
        oxygen_saturation = random.randint(*ABNORMAL_OXYGEN_SATURATION_RANGE)
    
    # Generate movement data once in a while
    movement = random.randint(0, 3) if random.random() < 0.5 else MOVEMENT  # 20% chance of generating new movement data

    # Preprocess the data
    data = preprocess_data([temperature, respiration_rate, oxygen_saturation, movement])
    
    # Make prediction using the model
    prediction = model.predict(data)
    
    # Update the model with new data if threshold is reached
    if data_counter >= MODEL_UPDATE_THRESHOLD:
        label = 0  # Example label (replace this with your actual label)
        update_model(model, data, np.array([label]))
        data_counter = 0
    else:
        data_counter += 1
    
    return jsonify(temperature=temperature, respiration_rate=respiration_rate, oxygen_saturation=oxygen_saturation, movement=movement, prediction=float(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
