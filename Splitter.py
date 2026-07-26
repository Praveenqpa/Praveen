import os
import sys
import tkinter as tk
from tkinter import filedialog
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


def split_audio(file_path, output_dir="./output"):
    if not os.path.exists(file_path):
        print(f"Error: Could not find file at '{file_path}'")
        return

    print("--- Starting AI Audio Stem Splitter ---")
    
    # 1. Detect hardware acceleration (GPU vs CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using processing device: {device.upper()}")

    # 2. Load Meta's pretrained Demucs model
    print("Loading AI model (htdemucs)...")
    try:
        model = get_model('htdemucs')
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load Demucs model: {e}")
        return

    # 3. Load audio file into tensor
    print(f"Loading audio file: {os.path.basename(file_path)}...")
    try:
        wav, sr = torchaudio.load(file_path)
        wav = wav.to(device)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        print("Tip: If reading MP3 fails, run 'pip install ffmpeg-python'")
        return

    # 4. Perform Separation
    print("Separating stems... (Please wait, this takes 1-3 minutes on CPU)")
    with torch.no_grad():
        # Shape output: [4, channels, samples]
        sources = apply_model(model, wav.unsqueeze(0), progress=True)[0]

    # 5. Save separated WAV files
    song_name = os.path.splitext(os.path.basename(file_path))[0]
    save_folder = os.path.join(output_dir, song_name)
    os.makedirs(save_folder, exist_ok=True)

    stems = ['drums', 'bass', 'other', 'vocals']
    print("\nSaving separated tracks...")
    
    for i, stem in enumerate(stems):
        out_path = os.path.join(save_folder, f"{stem}.wav")
        # Move back to CPU before saving
        torchaudio.save(out_path, sources[i].cpu(), sr)
        print(f" Saved: {out_path}")

    print(f"\n✨ Done! Stems saved in folder: {os.path.abspath(save_folder)}")


if __name__ == "__main__":
    # Open native file dialog to choose song
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("Please select an audio file from the popup window...")
    selected_file = filedialog.askopenfilename(
        title="Select Song to Split",
        filetypes=[("Audio Files", "*.mp3 *.wav *.flac *.m4a *.ogg")]
    )

    if selected_file:
        split_audio(selected_file)
    else:
        print("No file selected. Exiting.")
