import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

# ----------------------------
# 1. Data Loading & Preparation (Regression)
# ----------------------------
def load_and_prepare_data(filepath):
    """Load and preprocess the accelerometer data for regression"""
    df = pd.read_csv(filepath)
    X = df[['x', 'y', 'z']].values
    y = df['pctid'].values  # Regression target (continuous value)
    return X, y

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
# 2. Regression Model Architectures
# ----------------------------
def build_model_1(input_shape):
    """Simple regression model with 1 hidden layer"""
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=input_shape),
        layers.Dense(1)  # Single output for regression
    ])
    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae'])  # Mean Absolute Error
    return model

def build_model_2(input_shape):
    """Medium regression model with 2 hidden layers"""
    model = keras.Sequential([
        layers.Dense(64, activation='tanh', input_shape=input_shape),
        layers.Dense(32, activation='tanh'),
        layers.Dropout(0.1),
        layers.Dense(1)  # Single output for regression
    ])
    
    # Proper way to use Huber loss
    model.compile(optimizer='rmsprop',
                loss=keras.losses.Huber(),  # Huber loss implementation
                metrics=['mae'])
    return model

def build_model_3(input_shape):
    """Complex regression model with 3 hidden layers"""
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape,
                    kernel_regularizer=keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_absolute_error',
                metrics=['mse'])  # Mean Squared Error
    return model

# ----------------------------
# 3. Training & Evaluation (Regression)
# ----------------------------
def train_and_plot(model, X_train, y_train, X_val, y_val, epochs, batch_size, model_name):
    """Train regression model and plot metrics"""
    print(f"\nTraining {model_name}")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Create individual plots
    plt.figure(figsize=(12, 5))
    
    # MAE/MSE plot
    plt.subplot(1, 2, 1)
    metric_name = list(history.history.keys())[1]  # Get first metric name
    plt.plot(history.history[metric_name], label=f'Train {metric_name.upper()}')
    plt.plot(history.history[f'val_{metric_name}'], 
             label=f'Validation {metric_name.upper()}', linestyle='--')
    plt.title(f'{model_name}\n{metric_name.upper()} vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.upper())
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
# Main Execution (Regression)
# ----------------------------
def main():
    # 1. Load and prepare data
    X, y = load_and_prepare_data("accelerometer.csv")
    X_train, y_train, X_val, y_val, X_test, y_test = create_splits(X, y)
    
    # 2. Initialize regression models
    input_shape = (X_train.shape[1],)
    models = {
        "Model 1 (1 Hidden Layer)": build_model_1(input_shape),
        "Model 2 (2 Hidden Layers)": build_model_2(input_shape),
        "Model 3 (3 Hidden Layers)": build_model_3(input_shape)
    }
    
    # 3. Train models with different settings
    histories = {}
    for name, model in models.items():
        epochs = 100 if "Model 3" in name else 50  # Longer training for complex model
        batch_size = 64 if "Model 3" in name else 32
        histories[name] = train_and_plot(
            model, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, model_name=name
        )
    
    # 4. Evaluate on test set
    print("\nTest Set Performance:")
    for name, model in models.items():
        test_loss, test_metric = model.evaluate(X_test, y_test, verbose=0)
        print(f"{name}:")
        print(f"  Loss = {test_loss:.4f}")
        print(f"  Metric = {test_metric:.4f}\n")

if __name__ == "__main__":
    main()