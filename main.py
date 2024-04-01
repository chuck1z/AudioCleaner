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

st.title("Basically a better Adobe's Enhance Speech")
st.subheader("Made possible thanks to DeepFilterNet")

uploaded_file = st.file_uploader("Upload your audio file here")
if uploaded_file is not None:
    audio_array, sample_rate = librosa.load(uploaded_file, sr=None) 
    audio_stream = io.BytesIO(uploaded_file.getvalue())
    audio = pydub.AudioSegment.from_file(audio_stream)
    temp_audio_file = io.BytesIO()
    audio.export(temp_audio_file, format="wav")
    raw_audio = temp_audio_file.getvalue()
    st.audio(raw_audio, format='audio/wav')
    
    # print(waveform)
    if st.button('Clean Audio'):
        audio_stream = io.BytesIO(raw_audio) 
        model, df_state, _ = init_df()
        waveform, sample_rate = torchaudio.load(audio_stream)
        enhanced = enhance(model, df_state, waveform)
        enhanced_numpy = enhanced.cpu().numpy()
        st.write('Cleaned audio')
        st.audio(enhanced_numpy, format='audio/wav', sample_rate=sample_rate)

        # Plot spectrograms
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Original audio spectrogram
        orig_spec = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(orig_spec, ref=np.max), sr=sample_rate, x_axis='time', y_axis='mel', ax=axs[0])
        axs[0].set(title='Original Audio Spectrogram')

        # Enhanced audio spectrogram
        enhanced_spec = librosa.feature.melspectrogram(y=enhanced_numpy.squeeze(), sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(enhanced_spec, ref=np.max), sr=sample_rate, x_axis='time', y_axis='mel', ax=axs[1])
        axs[1].set(title='Enhanced Audio Spectrogram')

        st.pyplot(fig)

        st.write(f"DeepFilterNet's GitHub repo: <a href='https://github.com/Rikorose/DeepFilterNet'>https://github.com/Rikorose/DeepFilterNet</a>", unsafe_allow_html=True)
        # st.audio(uploaded_file.getvalue(), format='audio/wav')


# debugging archive
# audio_path = download_file(
#         "https://github.com/Rikorose/DeepFilterNet/raw/e031053/assets/noisy_snr0.wav",
#         download_dir=".",
#     )
# st.audio(audio_path, format='audio/wav')

# model, df_state, _ = init_df()
# audio, _ = load_audio(audio_path, sr=df_state.sr())
# waveform, sample_rate = torchaudio.load('noisy_snr0.wav')

# print(audio)
# st.write(audio)
# print(waveform)
# st.write(waveform)

# # Denoise the audio
# enhanced = enhance(model, df_state, audio)
# # Save for listening
# # save_audio("enhanced.wav", enhanced2, df_state.sr())
# enhanced_numpy = enhanced.cpu().numpy()
# sample_rate = df_state.sr()
# st.audio(enhanced_numpy, format='audio/wav', sample_rate=sample_rate)