# Adobe Enhance Speech Alternative with DeepFilterNet
This Streamlit web application provides a user-friendly interface for high-quality audio noise reduction powered by the DeepFilterNet framework.

## Features
* Simple Audio Upload: Drag and drop your audio files (WAV, MP3, etc.) directly into the web app.</br>
* Audio Playback: Preview both the original audio and the cleaned version.</br>
* Spectrogram Visualizations: Compare the before and after spectrograms to visually understand the noise reduction process.</br>
* Powered by DeepFilterNet: Leverages the robust deep learning capabilities of DeepFilterNet for superior results.</br>

## How to Use
1. Upload your audio file.
2. Preview the original audio.
3. Click the 'Clean Audio' button.
4. Listen to the enhanced audio.
5. Compare spectrograms for a deeper look at the results.

## Installation
Ensure you have the following prerequisites:
- Python >= 3.9
- Streamlit
- PyTorch
- DeepFilterNet

Run
```bash
git clone https://github.com/chuck1z/AudioCleaner
cd AudioCleaner
pip install -r requirements.txt 
streamlit run main.py
```

## About DeepFilterNet
DeepFilterNet is an advanced deep learning framework for real-time speech enhancement. To learn more about this powerful tool, visit the GitHub repository: https://github.com/Rikorose/DeepFilterNet


## Contact/Contribution
Feel free to submit issues or feature requests on this project's GitHub repository. Contributions are welcome!
