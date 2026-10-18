import sys
import types
import io
import os
import tempfile
import subprocess
import streamlit as st
import numpy as np
import pydub
import soundfile as sf
import torchaudio
import matplotlib.pyplot as plt
import librosa
import torch

# --- Backward compatibility shim for DeepFilterNet with modern torchaudio ---
if "torchaudio.backend.common" not in sys.modules:
    backend_module = types.ModuleType("torchaudio.backend")
    common_module = types.ModuleType("torchaudio.backend.common")
    common_module.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
    backend_module.common = common_module
    sys.modules["torchaudio.backend"] = backend_module
    sys.modules["torchaudio.backend.common"] = common_module

from df.enhance import enhance, init_df, save_audio
from df.io import resample

st.title("Probably(?) a better Adobe Enhance Speech")
st.subheader("Made possible thanks to DeepFilterNet")

uploaded_file = st.file_uploader("Upload your audio file here", type=["mp3", "m4a", "wav", "flac", "ogg", "mov"])

if uploaded_file is not None:
    is_video = uploaded_file.name.lower().endswith('.mov')
    
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
        # Audio loading with soundfile fallback for compatibility across torchaudio versions
        try:
            waveform, sample_rate = torchaudio.load(audio_stream)
        except Exception:
            audio_stream.seek(0)
            data, sample_rate = sf.read(audio_stream, dtype='float32')
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

        if sample_rate != df_state.sr():
            waveform = resample(waveform, sample_rate, df_state.sr())
            sample_rate = df_state.sr()

        enhanced = enhance(model, df_state, waveform)
        enhanced_numpy = enhanced.cpu().numpy()
        st.write('Cleaned audio')
        st.audio(enhanced_numpy, format='audio/wav', sample_rate=sample_rate)

        # Save enhanced audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as enhanced_audio_file:
            sf.write(enhanced_audio_file.name, enhanced_numpy.T, sample_rate)
            enhanced_audio_path = enhanced_audio_file.name

        # If it's a video file, combine with original video
        if is_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mov") as temp_video_file:
                temp_video_file.write(uploaded_file.getvalue())
                temp_video_path = temp_video_file.name

            # Create output video with cleaned audio
            output_video_path = "cleaned_output.mp4"
            subprocess.run([
                'ffmpeg', '-i', temp_video_path, '-i', enhanced_audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                output_video_path
            ])

            # Provide download button for the video
            with open(output_video_path, 'rb') as f:
                st.download_button(
                    label="Download Cleaned Video",
                    data=f,
                    file_name="cleaned_output.mp4",
                    mime="video/mp4"
                )
            
            # Clean up temporary files
            os.unlink(temp_video_path)
            os.unlink(output_video_path)
        else:
            # Provide download button for audio only
            with open(enhanced_audio_path, 'rb') as f:
                st.download_button(
                    label="Download Cleaned Audio",
                    data=f,
                    file_name="cleaned_audio.wav",
                    mime="audio/wav"
                )

        # Clean up temporary files
        os.unlink(enhanced_audio_path)
        os.unlink(temp_audio_file.name)

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