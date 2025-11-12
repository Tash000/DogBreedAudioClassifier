import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model as keras_save_model
import joblib

def extract_features(file_path, duration=3, offset=0.5):
    try:
        y, sr = librosa.load(file_path, duration=duration, offset=offset)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_dir):
    X, Y = [], []
    class_labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for label in class_labels:
        folder = os.path.join(data_dir, label)
        for file in os.listdir(folder):
            if file.endswith(".wav"):
                file_path = os.path.join(folder, file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    Y.append(label)
    
    return np.array(X), np.array(Y), class_labels

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(
    data_dir,
    model_name="dense_bark_model",
    save_model_path=None,
    save_encoder_path=None,
    save_format="h5",
    plot_stats=True
):
    X, Y, class_labels = load_data(data_dir)

    le = LabelEncoder()
    Y_encoded = to_categorical(le.fit_transform(Y))

    if save_encoder_path is None:
        save_encoder_path = os.path.join("models", f"{model_name}_label_encoder.pkl")
    joblib.dump(le, save_encoder_path)

    X_train, X_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42)

    model = build_model(input_shape=X.shape[1], num_classes=Y_encoded.shape[1])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

    if save_model_path is None:
        save_model_path = os.path.join("models", f"{model_name}.h5")

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    if save_format == "keras":
        keras_save_model(model, save_model_path, save_format="keras")
    else:
        model.save(save_model_path)

    if plot_stats:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return model, le, class_labels, history
