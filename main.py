import streamlit as st
import numpy as np
import io
import pydub
from df.enhance import enhance, init_df, save_audio
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
import librosa
import torch
from torchaudio import AudioMetaData
import tempfile

st.title("Probably(?) a better Adobe Enhance Speech")
st.subheader("Made possible thanks to DeepFilterNet")

uploaded_file = st.file_uploader("Upload your audio file here", type=["mp3", "m4a", "wav", "flac", "ogg"])

if uploaded_file is not None:
    # Convert to WAV format
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio = pydub.AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()))
        audio.export(temp_audio_file.name, format="wav")
        temp_audio_file.seek(0)
        raw_audio = temp_audio_file.read()

    st.audio(raw_audio, format='audio/wav')

    if st.button('Clean Audio'):
        audio_stream = io.BytesIO(raw_audio)
        model, df_state, _ = init_df()
        waveform, sample_rate = torchaudio.load(audio_stream, backend='soundfile')
        enhanced = enhance(model, df_state, waveform)
        enhanced_numpy = enhanced.cpu().numpy()
        st.write('Cleaned audio')
        st.audio(enhanced_numpy, format='audio/wav', sample_rate=sample_rate)

        # Plot spectrograms
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Original audio spectrogram
        orig_spec = librosa.feature.melspectrogram(y=waveform.squeeze().numpy(), sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(orig_spec, ref=np.max), sr=sample_rate, x_axis='time', y_axis='mel', ax=axs[0])
        axs[0].set(title='Original Audio Spectrogram')

        # Enhanced audio spectrogram
        enhanced_spec = librosa.feature.melspectrogram(y=enhanced_numpy.squeeze(), sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(enhanced_spec, ref=np.max), sr=sample_rate, x_axis='time', y_axis='mel', ax=axs[1])
        axs[1].set(title='Cleaned Audio Spectrogram')

        st.pyplot(fig)

        st.write(f"DeepFilterNet's GitHub repo: <a href='https://github.com/Rikorose/DeepFilterNet'>https://github.com/Rikorose/DeepFilterNet</a>", unsafe_allow_html=True)