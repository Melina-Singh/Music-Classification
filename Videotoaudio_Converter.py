from moviepy.editor import VideoFileClip
import os

def convert_videos_to_audios(base_directory, output_directory):
    try:
        # Iterate through each folder in the base directory
        for folder_name in os.listdir(base_directory):
            folder_path = os.path.join(base_directory, folder_name)
            
            # Skip if it's not a directory
            if not os.path.isdir(folder_path):
                continue
            
            # Create corresponding output folder if not exists
            output_folder_path = os.path.join(output_directory, folder_name)
            os.makedirs(output_folder_path, exist_ok=True)
            
            # Iterate through each file in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                
                # Check if it's a video file
                if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mkv"):
                    # Load the video file
                    video = VideoFileClip(file_path)
                    
                    # Extract audio from the video
                    audio = video.audio
                    
                    # Define output audio file path
                    output_audio_file = os.path.join(output_folder_path, os.path.splitext(filename)[0] + ".mp3")
                    
                    # Write the audio to a new file
                    audio.write_audiofile(output_audio_file)
                    
                    print(f"Conversion successful: {output_audio_file}")
                    
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
base_directory = "D:\\Youtube\\Videos"
output_directory = "D:\\Music Classification\\Audios"
convert_videos_to_audios(base_directory, output_directory)
