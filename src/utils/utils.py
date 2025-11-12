import os
import yt_dlp
from pydub import AudioSegment
import librosa
import numpy as np
import uuid
from datetime import datetime
from tensorflow.keras.models import save_model, load_model

# --- Project path configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========== 1. Download Audio from YouTube ========== #
def download_audio(youtube_url, output_dir, filename_prefix=None):
    os.makedirs(output_dir, exist_ok=True)
    filename_prefix = filename_prefix or str(uuid.uuid4())
    output_path = os.path.join(output_dir, f"{filename_prefix}.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        downloaded_file = output_path.replace('%(ext)s', 'wav')
        return downloaded_file

# ========== 2. Split Audio into 3-sec Clips ========== #
def split_audio(audio_path, breed_label, base_dir=None, clip_duration=3, for_test=False):
    if base_dir is None:
        base_dir = os.path.join(DATA_DIR, "dataset", "clips")
    target_dir = os.path.join(DATA_DIR, "test_clips") if for_test else base_dir
    breed_dir = os.path.join(target_dir, breed_label)
    os.makedirs(breed_dir, exist_ok=True)

    audio = AudioSegment.from_wav(audio_path)
    total_duration = len(audio)
    num_clips = total_duration // (clip_duration * 1000)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    clip_paths = []

    for i in range(num_clips):
        start = i * clip_duration * 1000
        end = start + clip_duration * 1000
        clip = audio[start:end]
        clip_filename = f"{breed_label}_{timestamp}_clip_{i}.wav"
        full_path = os.path.join(breed_dir, clip_filename)
        clip.export(full_path, format="wav")
        clip_paths.append(full_path)

    return clip_paths

# ========== 3. Extract MFCC Feature from Audio Clip ========== #
def extract_features(file_path, duration=3.0, offset=0.5, n_mfcc=40):
    try:
        y, sr = librosa.load(file_path, duration=duration, offset=offset)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {file_path}: {e}")
        return None

# ========== 4. Load Dataset from Breed Folders ========== #
def load_data_from_folders(base_dir, class_labels):
    X, Y = [], []
    for label in class_labels:
        label_path = os.path.join(base_dir, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(".wav"):
                clip_path = os.path.join(label_path, file)
                features = extract_features(clip_path)
                if features is not None:
                    X.append(features)
                    Y.append(label)
    return np.array(X), np.array(Y)

# ========== 5. Predict the Breed of a New Audio Clip ========== #
def predict_audio_clip(file_path, model, label_encoder, confidence_threshold=0.5):
    feature = extract_features(file_path)
    if feature is None:
        return "Feature Extraction Failed", 0.0

    prediction = model.predict(np.expand_dims(feature, axis=0))[0]
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(prediction[predicted_index])

    if confidence < confidence_threshold:
        return "Unknown", confidence
    return predicted_label, confidence

# ========== 6. Get Top-K Predictions ========== #
def predict_top_k(file_path, model, label_encoder, k=3):
    feature = extract_features(file_path)
    if feature is None:
        return []

    prediction = model.predict(np.expand_dims(feature, axis=0))[0]
    top_indices = prediction.argsort()[-k:][::-1]
    return [(label_encoder.inverse_transform([i])[0], float(prediction[i])) for i in top_indices]

# ========== 7. Save model in .h5 or .keras format ========== #
def save_model_format(model, save_path, use_keras_format=False):
    if use_keras_format:
        save_model(model, save_path, save_format='keras')
    else:
        model.save(save_path)

# ========== 8. Load available models ========== #
def list_models(models_dir=None):
    if models_dir is None:
        models_dir = MODEL_DIR
    return [f for f in os.listdir(models_dir) if f.endswith(".h5") or f.endswith(".keras")]

# ========== 9. Load model by name ========== #
def load_named_model(model_name, models_dir=None):
    if models_dir is None:
        models_dir = MODEL_DIR
    model_path = os.path.join(models_dir, model_name)
    return load_model(model_path)
