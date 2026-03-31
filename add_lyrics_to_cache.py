"""
Smart Lyrics Extractor.
Pulls text directly from Sonics CSVs if available, otherwise runs Whisper.
Generates SBERT embeddings and injects them into the merged feature cache.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress FP16 warnings from Whisper on CPU
warnings.filterwarnings("ignore")

from feature_extraction.lyrics import extract_lyrics, extract_lyric_embedding, unload_models

import argparse

DATA_DIR = Path("data")

def build_sonics_lyric_dictionary() -> dict:
    """
    Reads all Sonics CSVs and creates a mapping of filename -> lyrics.
    This saves us from running Whisper on 97k+ files!
    """
    print("Building Sonics lyric dictionary from CSVs...")
    lyric_map = {}
    
    # List of all Sonics metadata files that contain a 'lyrics' column
    sonics_csvs = [
        "sonics/pair_fake.csv",
        "sonics/pair_real.csv",
        "sonics/sibling_fake.csv",
        "sonics/extra_fake_metadata.csv",
        "sonics/extra_real_metadata.csv"
    ]
    
    for csv_path in sonics_csvs:
        full_path = DATA_DIR / csv_path
        if full_path.exists():
            df = pd.read_csv(full_path, low_memory=False)
            for _, row in df.iterrows():
                if "lyrics" in row and pd.notna(row["lyrics"]):
                    # Handle real tracks that are named by youtube_id
                    if "youtube_id" in row and pd.notna(row["youtube_id"]):
                        key = f"{row['youtube_id']}.wav"
                    else:
                        key = row["filename"]
                        if not key.endswith(".wav"):
                            key += ".wav"
                    
                    lyric_map[key] = str(row["lyrics"])
                    
    print(f"Loaded {len(lyric_map)} lyrics from Sonics metadata.")
    return lyric_map


def main():
    parser = argparse.ArgumentParser(description="Add lyric embeddings to feature cache")
    parser.add_argument("--cache_dir", type=str, default="feature_cache_cpu",
                        help="Path to the feature cache directory")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "feature_metadata.csv"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found. Did you run merge_caches.py?")
        return

    df_meta = pd.read_csv(meta_path)
    lyric_map = build_sonics_lyric_dictionary()

    print(f"Processing {len(df_meta)} tracks for lyric embeddings...")

    # We keep track of stats to print at the end
    stats = {"skipped_existing": 0, "from_csv": 0, "from_whisper": 0, "failed": 0}

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        cache_path = cache_dir / Path(row["cache_path"]).name
        audio_path = row["audio_path"]

        if not cache_path.exists():
            continue

        # 1. Load the existing features for this track
        data = dict(np.load(cache_path, allow_pickle=True))
        
        # 2. Skip if we already successfully processed this file before
        if "lyric_embedding" in data:
            stats["skipped_existing"] += 1
            continue

        try:
            text = ""
            # 3. FAST PATH: Is it a Sonics track? Grab from dictionary.
            if row["dataset"] == "sonics":
                filename = Path(audio_path).name
                if filename in lyric_map:
                    text = lyric_map[filename]
                    stats["from_csv"] += 1
            
            # 4. SLOW PATH: Not Sonics, or missing from dictionary. Run Whisper.
            if not text:
                if Path(audio_path).exists():
                    # model_size="base" is a good balance of speed and accuracy
                    text = extract_lyrics(audio_path, model_size="base")
                    stats["from_whisper"] += 1
                else:
                    raise FileNotFoundError("Audio file missing.")

            # 5. Convert text to SBERT array
            embedding = extract_lyric_embedding(text)
            
            # 6. Update dictionary
            data["lyric_embedding"] = embedding
            data["lyrics_text"] = np.array(text) # Saving text just in case

        except Exception as e:
            # If Whisper fails or audio is corrupted, fail gracefully with zeros
            data["lyric_embedding"] = np.zeros(384, dtype=np.float32)
            stats["failed"] += 1

        # 7. Save the updated dictionary back to the .npz file
        np.savez_compressed(cache_path, **data)

    unload_models()
    
    print("\n=== LYRICS PROCESSING COMPLETE ===")
    print(f"Already had embeddings: {stats['skipped_existing']}")
    print(f"Instantly loaded from CSV: {stats['from_csv']}")
    print(f"Transcribed via Whisper: {stats['from_whisper']}")
    print(f"Failed/Instrumental: {stats['failed']}")

if __name__ == "__main__":
    main()