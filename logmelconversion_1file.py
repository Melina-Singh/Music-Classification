import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Function to extract log-Mel spectrogram from an audio file
def extract_log_mel_spectrogram(file_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(file_path, sr=sr)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return log_mel_spect

# Function to save a spectrogram as an image
def save_spectrogram_image(spect, save_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect, sr=22050, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Path to the input audio file and the output image file
input_audio_path = r"D:\Music Classification\Audios\Doteli\1_chunk1.wav"
output_image_path = r"D:\Music Classification\Audios\spectrogram"

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# Extract log-Mel spectrogram
log_mel_spect = extract_log_mel_spectrogram(input_audio_path)

# Save the spectrogram as an image
save_spectrogram_image(log_mel_spect, output_image_path)

print(f'Log-Mel spectrogram saved to {output_image_path}')
