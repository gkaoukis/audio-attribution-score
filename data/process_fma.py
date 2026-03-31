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
MAX_SAMPLES_PER_MODEL = None 

def process_audio(input_path, output_path):
    """
    Reads an audio file, converts it to 16kHz Mono, and saves it as a 16-bit PCM WAV.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
        sf.write(output_path, y, sr, subtype=FORMAT_SUBTYPE)
        return True
    except Exception as e:
        print(f"Failed to process {input_path}: {e}")
        return False

def get_safe_name(name):
    """Replicates the file saving logic from the download script."""
    return "".join([c for c in str(name) if c.isalpha() or c.isdigit() or c in " -_,."]).rstrip()

def main():
    print(f"Loading manifest from {MANIFEST_PATH}...")
    df = pd.read_csv(MANIFEST_PATH, low_memory=False)

    # ==========================================
    # 1. Dataset Splitting
    # ==========================================
    print("Creating train/test splits (grouped by original track)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df['original_audio']))
    
    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'

    train_originals = set(df[df['split'] == 'train']['original_audio'])
    test_originals = set(df[df['split'] == 'test']['original_audio'])
    assert len(train_originals.intersection(test_originals)) == 0, "Leakage detected!"

    # ==========================================
    # 2. Configurable Sampling
    # ==========================================
    if MAX_SAMPLES_PER_MODEL is not None:
        df = df.groupby(['split', 'generator'], group_keys=False).apply(
            lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_MODEL), random_state=RANDOM_STATE)
        )

    # ==========================================
    # 3. Process Generated Audio Files
    # ==========================================
    print(f"Processing Generated Audio files to 16kHz, mono, 16-bit PCM WAV...")
    new_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting Generated"):
        rel_path = str(row['path_in_dataset'])
        input_file = INPUT_DIR / rel_path
        rel_path_wav = Path(rel_path).with_suffix('.wav')
        output_file = OUTPUT_DIR / row['split'] / rel_path_wav
        
        if input_file.exists():
            if not output_file.exists():
                process_audio(input_file, output_file)
            new_paths.append(str(output_file.relative_to(OUTPUT_DIR)))
        else:
            new_paths.append(None)
            
    df['processed_path'] = new_paths

    # ==========================================
    # 4. Process Original FMA Audio Files
    # ==========================================
    print(f"Processing Original FMA files to 16kHz, mono, 16-bit PCM WAV...")
    unique_originals = df[['original_audio', 'split']].drop_duplicates()
    original_paths_map = {}
    
    for _, row in tqdm(unique_originals.iterrows(), total=len(unique_originals), desc="Converting Originals"):
        orig_name = row['original_audio']
        safe_name = get_safe_name(orig_name)
        
        input_file = INPUT_DIR / "original_fma_tracks" / f"{safe_name}.mp3"
        rel_out_path = Path(row['split']) / "original_fma_tracks" / f"{safe_name}.wav"
        output_file = OUTPUT_DIR / rel_out_path
        
        if input_file.exists():
            if not output_file.exists():
                process_audio(input_file, output_file)
            original_paths_map[orig_name] = str(rel_out_path)
        else:
            print(f"  [Missing Original] {input_file}")
            original_paths_map[orig_name] = None

    # Link originals directly back to the main dataframe
    df['processed_original_path'] = df['original_audio'].map(original_paths_map)
    
    # Drop rows where EITHER the generated file OR the original FMA file failed/is missing
    df = df.dropna(subset=['processed_path', 'processed_original_path'])

    # Extract just the originals for their own manifest
    originals_manifest = df[['original_audio', 'split', 'processed_original_path']].drop_duplicates()
    originals_manifest = originals_manifest.rename(columns={'processed_original_path': 'processed_path'})

    # ==========================================
    # 5. Save Manifests
    # ==========================================
    out_manifest_path = OUTPUT_DIR / "processed_dataset_manifest.csv"
    orig_manifest_path = OUTPUT_DIR / "processed_originals_manifest.csv"
    
    df.to_csv(out_manifest_path, index=False)
    originals_manifest.to_csv(orig_manifest_path, index=False)
    
    print(f"\nDone! Processed datasets are in '{OUTPUT_DIR}'")
    print(f"Main Manifest: {out_manifest_path}")
    print(f"Originals Manifest: {orig_manifest_path}")

if __name__ == "__main__":
    main()