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
