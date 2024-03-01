from flask import Flask, render_template, jsonify
import random

app = Flask(__name__, static_url_path='/static')

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
    
    return jsonify(temperature=temperature, respiration_rate=respiration_rate, oxygen_saturation=oxygen_saturation, movement=movement)

if __name__ == '__main__':
    app.run(debug=True)