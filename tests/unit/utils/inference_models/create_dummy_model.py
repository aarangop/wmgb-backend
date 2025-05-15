import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a simple model for testing


def create_dummy_model():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(5,)),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    # Create test_data directory if it doesn't exist
    test_data_dir = os.path.join('./tests/', 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)

    # Create and save the model
    model = create_dummy_model()
    model_path = os.path.join(test_data_dir, 'dummy_model.h5')
    model.save(model_path)

    print(f"Dummy model saved to {model_path}")
    print(f"Model summary:")
    model.summary()
