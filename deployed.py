import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# Path to your saved Keras model
MODEL_PATH = 'deepfake_detection.h5'

# Load model (for Keras .h5 model)
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# Extract log-mel spectrogram features with shape (128, 109, 1)
def extract_features(audio_path, sr=16000, n_mels=128, max_frames=109):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)

    # Pad or truncate to fixed number of frames
    if log_mel_spec.shape[1] < max_frames:
        pad_width = max_frames - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel_spec = log_mel_spec[:, :max_frames]

    # Add channel dimension
    log_mel_spec = log_mel_spec[..., np.newaxis]
    return log_mel_spec[np.newaxis, ...]  # shape: (1, 128, 109, 1)

# Streamlit app
st.title('Audio Spoof Detection (SPS PROJECT)')
st.write('Upload an audio file to check if it is Real or Spoofed')

uploaded_file = st.file_uploader('Choose an audio file', type=['wav','mp3','flac'])
if uploaded_file is not None:
    # Save to a temporary file
    with open('temp_audio', 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.audio(uploaded_file, format='audio/wav')

    # Extract features and predict
    with st.spinner('Extracting features and predicting...'):
        feats = extract_features('temp_audio')
        pred = model.predict(feats)
        label = 'Real' if pred[0][0] > 0.5 else 'Spoof'

    st.success(f'The audio is predicted as: **{label}**')

# Deployment instructions
st.sidebar.title('Instructions')
st.sidebar.write(
    '''
1. Place your trained model at `model.h5` in the same folder.
2. Run: `streamlit run streamlit_app.py`
3. To deploy, push this folder to your favorite hosting (e.g., Streamlit Cloud, Heroku).
    ''')
