import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import joblib

# 1. Extract full MFCC matrix (no averaging)
def extract_features_cnn(file_path, duration=3, offset=0.5, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, duration=duration, offset=offset)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfccs.T  # Shape: (Time, 40)
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None

# 2. Load all audio clips and their MFCC matrices
def load_data_cnn(data_dir, n_mfcc=40):
    X, Y = [], []
    class_labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Detected classes: {class_labels}")

    for label in class_labels:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                path = os.path.join(folder, file)
                features = extract_features_cnn(path, n_mfcc=n_mfcc)
                if features is not None:
                    X.append(features)
                    Y.append(label)

    # Padding/truncating to fixed time steps
    max_len = 130
    padded_X = []
    for f in X:
        if f.shape[0] > max_len:
            padded = f[:max_len, :]
        else:
            padded = np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), mode='constant')
        padded_X.append(padded)

    return np.array(padded_X), np.array(Y), class_labels

# 3. CNN Model
def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 4. Train CNN Model
def train_cnn(data_dir, model_name="cnn_bark_model", encoder_path=None, plot_stats=True):
    print("[INFO] Loading data...")
    X, Y, class_labels = load_data_cnn(data_dir)

    le = LabelEncoder()
    Y_encoded = to_categorical(le.fit_transform(Y))

    if encoder_path is None:
        encoder_path = os.path.join("models", f"{model_name}_label_encoder.pkl")
    joblib.dump(le, encoder_path)

    X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

    print("[INFO] Building model...")
    model = build_cnn_model(input_shape=X.shape[1:], num_classes=Y_encoded.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    os.makedirs("models", exist_ok=True)

    if not model_name.endswith(('.h5', '.keras')):
        model_name += ".keras"  # default to keras if extension not given

    save_path = os.path.join("models", model_name)
    model.save(save_path)

    print(f"✅ Model saved: {save_path}")
    print(f"✅ Encoder saved: {encoder_path}")

    if plot_stats:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return model, le, class_labels, history
