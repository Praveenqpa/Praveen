import subprocess
import sys
import os

def split_audio(audio_path, output_dir="./output"):
    if not os.path.exists(audio_path):
        print(f"Error: File '{audio_path}' not found.")
        return

    print(f"Processing '{audio_path}' with Meta Demucs AI...")

    # Run Demucs CLI through Python
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "--out", output_dir,
        "-n", "htdemucs",  # Fast & accurate hybrid transformer model
        audio_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccess! Separated stems are saved in: {output_dir}/htdemucs/")
    except subprocess.CalledProcessError as e:
        print("An error occurred during separation:", e)

if __name__ == "__main__":
    # Change 'song.mp3' to your input audio file
    split_audio("song.mp3")
