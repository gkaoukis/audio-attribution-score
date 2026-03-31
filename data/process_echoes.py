import os
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_DIR = Path("echoes")
OUTPUT_DIR = Path("echoes_processed")
MANIFEST_PATH = INPUT_DIR / "dataset_manifest.csv"

# Audio parameters
TARGET_SR = 16000
FORMAT_SUBTYPE = 'PCM_16'

# Split & Sampling parameters
TEST_SIZE = 0.3
RANDOM_STATE = 42
# Set to an integer (e.g., 500) to balance models. Set to None to keep all data.
MAX_SAMPLES_PER_MODEL = None 

def process_audio(input_path, output_path):
    """
    Reads an audio file, converts it to 16kHz Mono, and saves it as a 16-bit PCM WAV.
    """
    try:
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # librosa.load naturally converts to mono (mono=True by default) and resamples
        y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
        
        # Write out as 16-bit PCM WAV
        sf.write(output_path, y, sr, subtype=FORMAT_SUBTYPE)
        return True
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")
        return False

def main():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    df = pd.read_csv(MANIFEST_PATH, low_memory=False)

    # ==========================================
    # 1. Dataset Splitting (No Data Leakage)
    # ==========================================
    print("Creating train/test splits (grouped by original track)...")
    
    # GroupShuffleSplit ensures that all rows sharing the same 'original_audio' 
    # value end up in the exact same split (either all in train, or all in test).
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df['original_audio']))
    
    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'

    # STRICT LEAKAGE CHECK
    train_originals = set(df[df['split'] == 'train']['original_audio'])
    test_originals = set(df[df['split'] == 'test']['original_audio'])
    leakage = train_originals.intersection(test_originals)
    
    assert len(leakage) == 0, f"CRITICAL ERROR: Data leakage detected! Overlapping tracks: {leakage}"
    print(f"Split successful! Train unique tracks: {len(train_originals)} | Test unique tracks: {len(test_originals)}")

    # ==========================================
    # 2. Configurable Sampling
    # ==========================================
    if MAX_SAMPLES_PER_MODEL is not None:
        print(f"Applying balanced sampling: max {MAX_SAMPLES_PER_MODEL} samples per generator per split...")
        # Group by the split and the model name, then sample up to MAX_SAMPLES_PER_MODEL
        df = df.groupby(['split', 'generator'], group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_MODEL), random_state=RANDOM_STATE)
        )

    # ==========================================
    # 3. Audio Preprocessing
    # ==========================================
    print(f"Processing audio files to 16kHz, mono, 16-bit PCM WAV...")
    new_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting Audio"):
        # Rel path: e.g., TTA/model/file.mp3 or original_fma_tracks/file.mp3
        rel_path = str(row['path_in_dataset'])
        input_file = INPUT_DIR / rel_path
        
        # Change output extension to .wav since mp3 doesn't support PCM
        rel_path_wav = Path(rel_path).with_suffix('.wav')
        
        # New structure: echoes_processed/{split}/{ATA|TTA|original_fma_tracks}/...
        output_file = OUTPUT_DIR / row['split'] / rel_path_wav
        
        if input_file.exists():
            if not output_file.exists():
                process_audio(input_file, output_file)
            
            # Store the new relative path for the updated manifest
            new_paths.append(str(output_file.relative_to(OUTPUT_DIR)))
        else:
            # Catch missing files silently or print warnings
            new_paths.append(None)
            
    # Update manifest with new paths and drop any missing files
    df['processed_path'] = new_paths
    df = df.dropna(subset=['processed_path'])

    # ==========================================
    # 4. Save New Manifest
    # ==========================================
    out_manifest_path = OUTPUT_DIR / "processed_dataset_manifest.csv"
    df.to_csv(out_manifest_path, index=False)
    print(f"\nDone! Processed dataset is in '{OUTPUT_DIR}'")
    print(f"Updated manifest saved to: {out_manifest_path}")

if __name__ == "__main__":
    main()