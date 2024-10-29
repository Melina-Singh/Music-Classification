from pydub import AudioSegment
from pydub.utils import make_chunks
import os

def split_audio(input_file, output_dir, chunk_length_ms=30000):
    try:
        # Load your audio file
        audio = AudioSegment.from_file(input_file)

        # Make chunks of `chunk_length_ms` milliseconds
        chunks = make_chunks(audio, chunk_length_ms)

        # Export all of the individual chunks as wav files
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]

        for i, chunk in enumerate(chunks):
            chunk_name = os.path.join(output_dir, f"{name_without_ext}_chunk{i+1}.wav")
            chunk.export(chunk_name, format="wav")
            print(f"Exported {chunk_name}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def split_all_files_in_directory(input_dir, output_dir, chunk_length_ms=30000):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp3"):  # Process only MP3 files
            input_file = os.path.join(input_dir, filename)
            split_audio(input_file, output_dir, chunk_length_ms)

# Example usage
input_dir = "D:\\5th Sem data\\Tamang"
output_dir = "D:\\Music Classification\\Data\\Tamangs"
split_all_files_in_directory(input_dir, output_dir)
