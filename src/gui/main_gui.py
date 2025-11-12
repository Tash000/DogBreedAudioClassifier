# main_gui.py — Full GUI for Dog Bark Audio Classification

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Combobox
import threading
import numpy as np
import matplotlib.pyplot as plt
import joblib
import librosa
from tensorflow.keras.models import load_model

from utils import download_audio, split_audio, extract_features
from train_model import train_model
from train_model_cnn import train_cnn

# === Config ===
AUDIO_DIR = "downloads"
SPLIT_DIR = "clips"
PREDICT_DIR = "predict"
MODEL_DIR = "models"
ENCODER_SUFFIX = "_label_encoder.pkl"
CONF_THRESHOLD = 0.4  # 40% threshold

for d in [AUDIO_DIR, SPLIT_DIR, PREDICT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

def get_supported_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(('.h5', '.keras'))]

def threaded(func):
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs).start()
    return wrapper

def extract_features_cnn(file_path, max_len=130, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
        if mfcc.shape[0] > max_len:
            mfcc = mfcc[:max_len, :]
        else:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
        return mfcc
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
        return None

class DogBreedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog Bark Breed Classifier")
        self.root.geometry("740x600")

        self.model = None
        self.label_encoder = None
        
        self.update_breed_dropdown()

        tk.Label(root, text="YouTube Link").pack()
        self.yt_entry = tk.Entry(root, width=70)
        self.yt_entry.pack(pady=5)

        tk.Label(root, text="Breed Label").pack()
        self.breed_var = tk.StringVar()
        self.breed_box = Combobox(root, textvariable=self.breed_var, width=30)
        self.refresh_breed_list()
        self.breed_box.pack(pady=5)

        self.breed_entry = tk.Entry(root, width=30)
        self.breed_entry.pack(pady=2)
        self.breed_entry.insert(0, "")

        tk.Button(root, text="📥 Download & Prepare (for Training)", command=self.download_and_prepare).pack(pady=5)
        tk.Button(root, text="🎯 Download for Prediction", command=self.download_for_prediction).pack(pady=5)

        tk.Label(root, text="Select Model Type").pack()
        self.model_type_box = Combobox(root, values=["Dense", "CNN"])
        self.model_type_box.set("Dense")
        self.model_type_box.pack(pady=5)

        tk.Label(root, text="Model Save Name").pack()
        self.model_name_entry = tk.Entry(root, width=30)
        self.model_name_entry.insert(0, "my_model")
        self.model_name_entry.pack(pady=5)

        tk.Button(root, text="🧠 Train Model", command=self.train).pack(pady=5)

        tk.Label(root, text="Select Model File").pack()
        self.model_file_box = Combobox(root, values=get_supported_models())
        self.model_file_box.set(get_supported_models()[0] if get_supported_models() else "")
        self.model_file_box.pack(pady=5)

        tk.Button(root, text="🔍 Predict from Audio File", command=self.predict_from_file).pack(pady=5)

        self.output_text = tk.Label(root, text="", fg="blue")
        self.output_text.pack(pady=10)
        
    def refresh_breed_list(self):
        if os.path.exists(SPLIT_DIR):
            breeds = [d for d in os.listdir(SPLIT_DIR) if os.path.isdir(os.path.join(SPLIT_DIR, d))]
            self.breed_box['values'] = breeds
            if breeds:
                self.breed_box.set(breeds[0])

    def update_breed_dropdown(self):
        try:
            breeds = [d for d in os.listdir(SPLIT_DIR) if os.path.isdir(os.path.join(SPLIT_DIR, d))]
            self.breed_box['values'] = breeds
            if breeds:
                self.breed_box.set(breeds[0])
        except Exception as e:
            print(f"[ERROR] Updating breed dropdown: {e}")

    @threaded
    def download_and_prepare(self):
        link = self.yt_entry.get().strip()
        breed = self.breed_entry.get().strip() or self.breed_box.get().strip()

        if not link or not breed:
            messagebox.showerror("Error", "Please enter a YouTube link and breed label.")
            return
        try:
            self.output_text.config(text="Downloading and splitting...")
            path = download_audio(link, os.path.join(AUDIO_DIR, breed))
            clips = split_audio(path, breed, base_dir=SPLIT_DIR)
            self.output_text.config(text=f"✅ Saved {len(clips)} clips for {breed}")
            self.update_breed_dropdown()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.output_text.config(text="❌ Failed")

    @threaded
    def download_for_prediction(self):
        link = self.yt_entry.get().strip()
        if not link:
            messagebox.showerror("Error", "Please enter a YouTube link.")
            return
        try:
            self.output_text.config(text="Downloading for prediction...")
            path = download_audio(link, output_dir=PREDICT_DIR)
            self.output_text.config(text=f"✅ Downloaded to {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.output_text.config(text="❌ Failed")

    @threaded
    def train(self):
        model_type = self.model_type_box.get().lower()
        custom_name = self.model_name_entry.get().strip()
        if not custom_name:
            messagebox.showerror("Error", "Please enter a model name.")
            return
        try:
            self.output_text.config(text="Training in progress...")
            if model_type == "cnn":
                self.model, self.label_encoder, _, history = train_cnn(
                    data_dir=SPLIT_DIR,
                    model_name=custom_name,
                    encoder_path=os.path.join(MODEL_DIR, f"{custom_name}.keras{ENCODER_SUFFIX}"),
                    plot_stats=True
                )
                model_file = f"{custom_name}.keras"
            else:
                self.model, self.label_encoder, _, history = train_model(
                    data_dir=SPLIT_DIR,
                    model_name=custom_name,
                    save_encoder_path=os.path.join(MODEL_DIR, f"{custom_name}.h5{ENCODER_SUFFIX}"),
                    save_format="h5",
                    plot_stats=True
                )
                model_file = f"{custom_name}.h5"

            self.output_text.config(text="✅ Training complete.")
            updated_models = get_supported_models()
            self.model_file_box['values'] = updated_models
            self.model_file_box.set(model_file)
            self.update_breed_dropdown()

        except Exception as e:
            messagebox.showerror("Training Error", str(e))
            self.output_text.config(text="❌ Training failed.")

    @threaded
    def predict_from_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not file_path:
            return

        model_file = self.model_file_box.get()
        if not model_file:
            messagebox.showerror("Error", "Please select a model file.")
            return

        model_path = os.path.join(MODEL_DIR, model_file)
        encoder_path = os.path.join(MODEL_DIR, model_file.split('.')[0] + ENCODER_SUFFIX)

        try:
            model = load_model(model_path)
            le = joblib.load(encoder_path)
        except Exception as e:
            messagebox.showerror("Model Load Error", str(e))
            return

        try:
            input_shape = model.input_shape
            if len(input_shape) == 3:
                # CNN model expects (batch, time, mfcc)
                feature = extract_features_cnn(file_path)
                if feature is None:
                    raise ValueError("Could not extract CNN features.")
                feature = np.expand_dims(feature, axis=0)
            elif len(input_shape) == 2:
                # Dense model expects (batch, 40)
                feature = extract_features(file_path)
                if feature is None:
                    raise ValueError("Could not extract Dense features.")
                feature = np.expand_dims(feature, axis=0)
            else:
                raise ValueError(f"Unsupported input shape: {input_shape}")

            pred = model.predict(feature)[0]
            confidence = np.max(pred)
            if confidence >= CONF_THRESHOLD:
                label = le.inverse_transform([np.argmax(pred)])[0]
            else:
                label = "Unknown"

            self.output_text.config(text=f"Prediction: {label} ({confidence*100:.2f}%)")

            # Plot confidence
            plt.bar(le.classes_, pred, color='skyblue')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.ylabel("Confidence")
            plt.title("Prediction Confidence")
            plt.tight_layout()
            plt.grid(True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.output_text.config(text="❌ Prediction failed.")

# === Entry Point ===
if __name__ == "__main__":
    root = tk.Tk()
    app = DogBreedApp(root)
    root.mainloop()
