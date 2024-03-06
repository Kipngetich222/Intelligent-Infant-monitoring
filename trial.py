import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define global variables
INPUT_SIZE = 4  # Temperature, Respiration, Oxygen Saturation, Movement

# Define the model
def build_nn_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(INPUT_SIZE,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to preprocess data
def preprocess_data(data):
    return np.array(data).reshape(1, -1)  # Reshape data into a format compatible with the model input

# Function to update the model with new data
def update_model(model, X, y):
    model.fit(X, y, epochs=1, verbose=0)

# Function to analyze trend and provide prediction and suggestions
def analyze_trend(data_history, current_data):
    # Assuming data_history is a list of tuples (temperature, respiration_rate, oxygen_saturation, movement)
    # Let's compute the average values for each parameter
    avg_temperature = np.mean([entry[0] for entry in data_history])
    avg_respiration_rate = np.mean([entry[1] for entry in data_history])
    avg_oxygen_saturation = np.mean([entry[2] for entry in data_history])
    avg_movement = np.mean([entry[3] for entry in data_history])
    
    # Compare current data with historical averages to determine trend
    # For simplicity, let's say if any parameter deviates significantly from the average, it's considered abnormal
    if current_data[0] < avg_temperature - 0.5 or current_data[0] > avg_temperature + 0.5 \
        or current_data[1] < avg_respiration_rate - 5 or current_data[1] > avg_respiration_rate + 5 \
        or current_data[2] < avg_oxygen_saturation - 5 or current_data[2] > avg_oxygen_saturation + 5 \
        or current_data[3] != avg_movement:
        prediction = "The infant's health condition is concerning."
        suggestion = "Please take the infant to the hospital for a thorough check-up."
    else:
        prediction = "The infant's health condition appears to be stable."
        suggestion = "Continue monitoring the infant closely and ensure they are comfortable."
    
    return prediction, suggestion
