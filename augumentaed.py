import torch
import librosa
from torchaudio.transforms import PitchShift
import random
import os

# Function to apply time stretching and pitch shifting
def augment_audio(waveform, sample_rate):
    augmented_waveform = waveform.copy()  # Create a copy of the NumPy array

    # Apply time stretching
    if random.random() < 0.5:  # Apply with a 50% chance
        stretch_factor = random.uniform(0.8, 1.2)
        augmented_waveform = librosa.effects.time_stretch(augmented_waveform, stretch_factor)
    
    # Apply pitch shifting
    if random.random() < 0.5:
        pitch_shift_steps = random.uniform(-2, 2)
        augmented_waveform = librosa.effects.pitch_shift(augmented_waveform, sample_rate, n_steps=pitch_shift_steps)
    
    return torch.tensor(augmented_waveform)

# Define input and output paths
input_folder = r"D:\Music Classification\Doteli"
output_folder = r"D:\Music Classification\Augmented_Data"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Augment audio files inside the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_folder, filename)
        waveform, sample_rate = librosa.load(input_path, sr=None, mono=True)
        augmented_waveform = augment_audio(waveform, sample_rate)
        output_path = os.path.join(output_folder, f"augmented_{filename}")
        librosa.output.write_wav(output_path, augmented_waveform.numpy(), sample_rate)
        print(f'Augmentation complete for "{filename}". Augmented file saved as "{output_path}".')

# Choose and augment three files from the original data and save them separately
original_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
random.shuffle(original_files)
for i, filename in enumerate(original_files[:3]):
    input_path = os.path.join(input_folder, filename)
    waveform, sample_rate = librosa.load(input_path, sr=None, mono=True)
    augmented_waveform = augment_audio(waveform, sample_rate)
    output_path = os.path.join(output_folder, f"original_augmented_{i+1}.wav")
    librosa.output.write_wav(output_path, augmented_waveform.numpy(), sample_rate)
    print(f'Augmentation complete for "{filename}". Augmented file saved as "{output_path}".')
