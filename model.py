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

import numpy as np

def analyze_trend(data_history, current_data):
    # Assuming data_history is a list of tuples (temperature, respiration_rate, oxygen_saturation, movement)
    avg_temperature = np.mean([entry[0] for entry in data_history])
    avg_respiration_rate = np.mean([entry[1] for entry in data_history])
    avg_oxygen_saturation = np.mean([entry[2] for entry in data_history])
    
    # Initialize prediction and suggestion
    issues = []
    suggestions = []
    
    # Temperature analysis
    if current_data[0] < 36.0:
        issues.append("low temperature")
        suggestions.append("ensure the infant is warm")
    elif current_data[0] > 37.5:
        issues.append("high temperature")
        suggestions.append("cool down the infant's surroundings and check for fever")
    
    # Respiration rate analysis
    if current_data[1] < 25:
        issues.append("low breathing rate")
        suggestions.append("ensure the infant's airway is clear and monitor closely")
    elif current_data[1] > 60:
        issues.append("high breathing rate")
        suggestions.append("check for signs of distress or fever and consider seeking medical advice")
    
    # Oxygen saturation analysis
    if current_data[2] < 92:
        issues.append("low oxygen saturation")
        suggestions.append("ensure the infant is in a well-ventilated area and consider seeking medical attention")
    
    # Movement analysis, assuming movement is a binary value where 1 indicates significant movement
    if current_data[3] == 1:
        issues.append("unexpected movement")
        suggestions.append("ensure the infant is safe and not in distress")
    
    # Formulate prediction and suggestion based on the identified issues
    if issues:
        prediction = "There may be issues with the infant's " + ", ".join(issues) + "."
        suggestion = "Consider to " + "; ".join(suggestions) + "."
    else:
        prediction = "The infant's health condition appears to be stable."
        suggestion = "Continue monitoring the infant closely and ensure they are comfortable."
    
    return prediction, suggestion

