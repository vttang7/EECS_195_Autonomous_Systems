import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, utils

# ----------------------------
# 1. Data Loading & Preparation
# ----------------------------
def load_and_prepare_data(filepath):
    """Load and preprocess the accelerometer data"""
    df = pd.read_csv(filepath)
    X = df[['x', 'y', 'z']].values
    y_raw = df['wconfid'].values
    num_classes = np.max(y_raw) + 1
    y = utils.to_categorical(y_raw, num_classes)
    return X, y, num_classes

def create_splits(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """Create train/validation/test splits"""
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    num_samples = len(X)
    train_end = int((1 - val_size - test_size) * num_samples)
    val_end = train_end + int(val_size * num_samples)
    
    return (X[indices[:train_end]], y[indices[:train_end]],
            X[indices[train_end:val_end]], y[indices[train_end:val_end]],
            X[indices[val_end:]], y[indices[val_end:]])

# ----------------------------
# 2. Model Architectures
# ----------------------------
def build_model_1(input_shape, num_classes):
    """Simple model with 1 hidden layer"""
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=input_shape),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def build_model_2(input_shape, num_classes):
    """Medium model with 2 hidden layers"""
    model = keras.Sequential([
        layers.Dense(64, activation='tanh', input_shape=input_shape),
        layers.Dense(32, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def build_model_3(input_shape, num_classes):
    """Complex model with 3 hidden layers"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape,
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.SGD(momentum=0.9),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

# ----------------------------
# 3. Training & Evaluation
# ----------------------------
def train_and_plot(model, X_train, y_train, X_val, y_val, epochs, batch_size, model_name):
    """Train model and plot its metrics separately"""
    print(f"\nTraining {model_name}")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Create individual plots for each model
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title(f'{model_name}\nAccuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title(f'{model_name}\nLoss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_metrics.png')
    plt.show()
    
    return history

# ----------------------------
# Main Execution
# ----------------------------
def main():
    # 1. Load and prepare data
    X, y, num_classes = load_and_prepare_data("accelerometer.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = create_splits(X, y)
    
    # 2. Initialize models
    input_shape = (X_train.shape[1],)
    models = {
        "Model 1 (1 Hidden Layer)": build_model_1(input_shape, num_classes),
        "Model 2 (2 Hidden Layers)": build_model_2(input_shape, num_classes),
        "Model 3 (3 Hidden Layers)": build_model_3(input_shape, num_classes)
    }
    
    # 3. Train models and plot separately
    histories = {}
    for name, model in models.items():
        epochs = 50 if "Model 3" in name else 30
        batch_size = 64 if "Model 3" in name else 32
        histories[name] = train_and_plot(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, model_name=name
        )
    
    # 4. Evaluate on test set
    print("\nTest Set Performance:")
    for name, model in models.items():
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"{name}:")
        print(f"  Accuracy = {test_acc:.4f}")
        print(f"  Loss = {test_loss:.4f}\n")

if __name__ == "__main__":
    main()