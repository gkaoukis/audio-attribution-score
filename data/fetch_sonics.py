"""Download and process SONICS dataset for Pairwise Similarity (Resume-Capable).

Usage:
    python fetch_sonics.py --data_dir ./data/sonics \
        --n_pairs 150 \
        --n_sibling_pairs 100 \
        --n_extra_fake 200 \
        --n_extra_real 150
"""
import argparse
import logging
import shutil
import zipfile
import re
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
import yt_dlp
from huggingface_hub import snapshot_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000


def parse_args():
    p = argparse.ArgumentParser(description="Process SONICS dataset for Similarity Pairs.")
    p.add_argument("--repo_id", type=str, default="awsaf49/sonics")
    p.add_argument("--data_dir", type=str, default="./sonics")
    p.add_argument("--n_pairs", type=int, default=100, help="Number of Real <-> Half Fake lyric pairs")
    p.add_argument("--n_sibling_pairs", type=int, default=100, help="Number of AI <-> AI sibling pairs (_0 and _1)")
    p.add_argument("--n_extra_fake", type=int, default=200, help="Number of extra diverse fake songs")
    p.add_argument("--n_extra_real", type=int, default=100, help="Number of extra real songs")
    p.add_argument("--max_workers", type=int, default=1, help="Parallel workers for YouTube downloads")
    return p.parse_args()


# ── Audio Utilities ───────────────────────────────────────────────────

def download_from_youtube(youtube_id: str, output_path: Path) -> bool:
    if output_path.exists(): return True
    try:
        with yt_dlp.YoutubeDL({
            "format": "bestaudio/best",
            "outtmpl": str(output_path.with_suffix("")),
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
            "postprocessor_args": ["-ar", str(TARGET_SR), "-ac", "1"],
            "quiet": True, "no_warnings": True, "nocheckcertificate": True,
        }) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
        return output_path.exists()
    except Exception as e:
        logger.error(f"Download failed for {youtube_id}: {e}")
        return False

def to_16k_mono(input_path: Path, output_path: Path) -> bool:
    if output_path.exists(): return True
    try:
        y, _ = librosa.load(str(input_path), sr=TARGET_SR, mono=True)
        sf.write(str(output_path), y, TARGET_SR, subtype="PCM_16")
        return True
    except Exception as e:
        logger.warning(f"Conversion failed for {input_path.name}: {e}")
        return False


# ── Logic Pipelines ───────────────────────────────────────────────────

def find_lyric_pairs(fake_df: pd.DataFrame, real_df: pd.DataFrame, n_needed: int) -> list:
    fake_filtered = fake_df[
        (fake_df["no_vocal"] == False) & 
        (fake_df["duration"] > 60) & 
        (fake_df["label"] == "half fake")
    ]
    real_v = real_df.dropna(subset=["lyrics"]).copy()
    fake_v = fake_filtered.dropna(subset=["lyrics"]).copy()
    
    if fake_v.empty or real_v.empty: return []

    logger.info("Computing TF-IDF lyric similarity for Pairs...")
    vec = TfidfVectorizer()
    real_tfidf = vec.fit_transform(real_v["lyrics"].astype(str))

    pairs, used_real_idx = [], set()
    for _, frow in tqdm(fake_v.iterrows(), total=len(fake_v), desc="Matching lyrics"):
        if len(pairs) >= n_needed: break
        
        fake_lyrics = str(frow["lyrics"])
        if len(fake_lyrics) < 50: continue

        sims = cosine_similarity(vec.transform([fake_lyrics]), real_tfidf)[0]
        for idx in sims.argsort()[::-1]:
            if sims[idx] < 0.9: break
            if idx in used_real_idx: continue
                
            real_lyrics = str(real_v.iloc[idx]["lyrics"])
            ratio = len(fake_lyrics) / max(len(real_lyrics), 1)
            
            if 0.9 <= ratio <= 1.1:
                pairs.append((frow, real_v.iloc[idx], sims[idx]))
                used_real_idx.add(idx)
                break
    return pairs

def find_sibling_fakes(fake_df: pd.DataFrame, n_needed: int, seed: int = 42):
    temp_df = fake_df.copy()
    temp_df['base_name'] = temp_df['filename'].astype(str).apply(lambda x: re.sub(r'_[01]$', '', x))
    
    counts = temp_df['base_name'].value_counts()
    valid_bases = counts[counts == 2].index.tolist()
    if not valid_bases: return []

    valid_df = temp_df[temp_df['base_name'].isin(valid_bases)]
    pair_metadata = valid_df.drop_duplicates(subset=['base_name'])
    
    try:
        sampled_bases = pair_metadata.groupby(['source', 'label'], group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, n_needed // 6)), random_state=seed)
        )
        if len(sampled_bases) > n_needed:
            sampled_bases = sampled_bases.sample(n=n_needed, random_state=seed)
        elif len(sampled_bases) < n_needed and len(pair_metadata) > len(sampled_bases):
            needed_extra = n_needed - len(sampled_bases)
            remaining = pair_metadata[~pair_metadata['base_name'].isin(sampled_bases['base_name'])]
            fillers = remaining.sample(n=min(needed_extra, len(remaining)), random_state=seed)
            sampled_bases = pd.concat([sampled_bases, fillers])
    except Exception as e:
        logger.warning(f"Stratification failed ({e}), falling back to random sampling.")
        sampled_bases = pair_metadata.sample(n=min(n_needed, len(pair_metadata)), random_state=seed)

    sibling_pairs = []
    for base in sampled_bases['base_name']:
        siblings = valid_df[valid_df['base_name'] == base]
        if len(siblings) == 2:
            sibling_pairs.append((siblings.iloc[0], siblings.iloc[1]))
            
    return sibling_pairs


# ── Main Execution ────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    fake_csv = data_dir / "fake_songs.csv"
    real_csv = data_dir / "real_songs.csv"
    fake_songs_dir = data_dir / "fake_songs"

    if not fake_csv.exists() or not real_csv.exists():
        logger.error("fake_songs.csv or real_songs.csv missing. Ensure metadata is downloaded.")
        return

    mp3_index = {p.stem: p for p in fake_songs_dir.glob("*.mp3")}
    logger.info(f"Indexed {len(mp3_index)} downloaded MP3s in fake_songs/")
    
    fake_df = pd.read_csv(fake_csv, low_memory=False)
    real_df = pd.read_csv(real_csv, low_memory=False)
    fake_df = fake_df[fake_df['filename'].isin(mp3_index.keys())]

    # Global tracking sets to prevent duplicate usage across runs
    used_fake_filenames = set()
    used_real_yt_ids = set()

    # --- 1. LYRIC PAIRS (Real <-> Half Fake) ---
    lyric_csv_path = data_dir / "lyric_pairs_mapping.csv"
    existing_lyric = 0
    if lyric_csv_path.exists():
        ex_df = pd.read_csv(lyric_csv_path)
        existing_lyric = len(ex_df)
        used_fake_filenames.update(ex_df['fake_filename'].astype(str).tolist())
        used_real_yt_ids.update(ex_df['real_youtube_id'].astype(str).tolist())

    n_lyric_needed = args.n_pairs - existing_lyric
    if n_lyric_needed > 0:
        logger.info(f"--- 1. Generating {n_lyric_needed} new Lyric Pairs (Found {existing_lyric} existing) ---")
        p_real_dir = data_dir / "pairs_real_16"; p_real_dir.mkdir(exist_ok=True)
        p_fake_dir = data_dir / "pairs_fake_16"; p_fake_dir.mkdir(exist_ok=True)
        
        available_fake = fake_df[~fake_df["filename"].isin(used_fake_filenames)]
        available_real = real_df[~real_df["youtube_id"].isin(used_real_yt_ids)]
        
        matched_lyric_pairs = find_lyric_pairs(available_fake, available_real, n_lyric_needed)
        
        pair_records = []
        for frow, rrow, sim in tqdm(matched_lyric_pairs, desc="Processing Lyric Pairs"):
            fname = str(frow["filename"])
            yt_id = str(rrow.get("youtube_id", "")).strip()
            
            real_wav = p_real_dir / f"{yt_id}.wav"
            fake_wav = p_fake_dir / f"{fname}.wav"

            if download_from_youtube(yt_id, real_wav):
                to_16k_mono(mp3_index[fname], fake_wav)
                pair_records.append({"fake_filename": fname, "real_youtube_id": yt_id, "similarity": round(sim, 4)})
                used_fake_filenames.add(fname)
                used_real_yt_ids.add(yt_id)
                
        if pair_records:
            pd.DataFrame(pair_records).to_csv(lyric_csv_path, mode='a', header=not lyric_csv_path.exists(), index=False)
    else:
        logger.info(f"--- 1. Lyric Pairs complete (Have {existing_lyric} >= Requested {args.n_pairs}) ---")

    # --- 2. SIBLING FAKE PAIRS (AI <-> AI) ---
    sib_csv_path = data_dir / "sibling_pairs_mapping.csv"
    existing_sib = 0
    if sib_csv_path.exists():
        ex_df = pd.read_csv(sib_csv_path)
        existing_sib = len(ex_df)
        used_fake_filenames.update(ex_df['sibling_0'].astype(str).tolist())
        used_fake_filenames.update(ex_df['sibling_1'].astype(str).tolist())

    n_sib_needed = args.n_sibling_pairs - existing_sib
    if n_sib_needed > 0:
        logger.info(f"--- 2. Generating {n_sib_needed} new Sibling Pairs (Found {existing_sib} existing) ---")
        sib_dir = data_dir / "extra_fake_16"; sib_dir.mkdir(exist_ok=True)
        
        available_fakes = fake_df[~fake_df["filename"].isin(used_fake_filenames)]
        sibling_pairs = find_sibling_fakes(available_fakes, n_sib_needed)
        
        sib_records = []
        for s1, s2 in tqdm(sibling_pairs, desc="Processing Sibling Pairs"):
            fn1, fn2 = str(s1["filename"]), str(s2["filename"])
            to_16k_mono(mp3_index[fn1], sib_dir / f"{fn1}.wav")
            to_16k_mono(mp3_index[fn2], sib_dir / f"{fn2}.wav")
            sib_records.append({"sibling_0": fn1, "sibling_1": fn2, "style": s1.get("style", "")})
            used_fake_filenames.update([fn1, fn2])
            
        if sib_records:
            pd.DataFrame(sib_records).to_csv(sib_csv_path, mode='a', header=not sib_csv_path.exists(), index=False)
    else:
        logger.info(f"--- 2. Sibling Pairs complete (Have {existing_sib} >= Requested {args.n_sibling_pairs}) ---")

    # --- 3. EXTRA DIVERSE FAKES ---
    ex_fake_csv_path = data_dir / "extra_fake_metadata.csv"
    existing_ex_fake = 0
    if ex_fake_csv_path.exists():
        ex_df = pd.read_csv(ex_fake_csv_path)
        existing_ex_fake = len(ex_df)
        used_fake_filenames.update(ex_df['filename'].astype(str).tolist())

    n_ex_fake_needed = args.n_extra_fake - existing_ex_fake
    if n_ex_fake_needed > 0:
        logger.info(f"--- 3. Sampling {n_ex_fake_needed} new Extra Fakes (Found {existing_ex_fake} existing) ---")
        extra_fake_dir = data_dir / "extra_fake_16"; extra_fake_dir.mkdir(exist_ok=True)
        
        available_fakes = fake_df[~fake_df["filename"].isin(used_fake_filenames)]
        try:
            extra_fakes = available_fakes.groupby(['source', 'label'], group_keys=False).apply(
                lambda x: x.sample(n=min(len(x), max(1, n_ex_fake_needed // 6)), random_state=42)
            ).head(n_ex_fake_needed)
        except Exception:
            extra_fakes = available_fakes.sample(n=min(n_ex_fake_needed, len(available_fakes)), random_state=42)
            
        for _, row in tqdm(extra_fakes.iterrows(), total=len(extra_fakes), desc="Converting Extra Fakes"):
            fname = str(row["filename"])
            to_16k_mono(mp3_index[fname], extra_fake_dir / f"{fname}.wav")
            used_fake_filenames.add(fname)
            
        if not extra_fakes.empty:
            extra_fakes.to_csv(ex_fake_csv_path, mode='a', header=not ex_fake_csv_path.exists(), index=False)
    else:
        logger.info(f"--- 3. Extra Fakes complete (Have {existing_ex_fake} >= Requested {args.n_extra_fake}) ---")

    # --- 4. EXTRA REAL SONGS ---
    ex_real_csv_path = data_dir / "extra_real_metadata.csv"
    existing_ex_real = 0
    if ex_real_csv_path.exists():
        ex_df = pd.read_csv(ex_real_csv_path)
        existing_ex_real = len(ex_df)
        used_real_yt_ids.update(ex_df['youtube_id'].astype(str).tolist())

    n_ex_real_needed = args.n_extra_real - existing_ex_real
    if n_ex_real_needed > 0:
        logger.info(f"--- 4. Sampling {n_ex_real_needed} new Extra Reals (Found {existing_ex_real} existing) ---")
        extra_real_dir = data_dir / "extra_real_16"; extra_real_dir.mkdir(exist_ok=True)
        
        available_reals = real_df[~real_df["youtube_id"].isin(used_real_yt_ids)]
        extra_reals = available_reals.sample(n=min(n_ex_real_needed, len(available_reals)), random_state=42)
        
        def _download_extra_real(row):
            yt_id = str(row.get("youtube_id", "")).strip()
            if yt_id: download_from_youtube(yt_id, extra_real_dir / f"{yt_id}.wav")

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            list(tqdm(pool.map(_download_extra_real, [r for _, r in extra_reals.iterrows()]), 
                      total=len(extra_reals), desc="Downloading Extra Reals"))
            
        if not extra_reals.empty:
            extra_reals.to_csv(ex_real_csv_path, mode='a', header=not ex_real_csv_path.exists(), index=False)
    else:
        logger.info(f"--- 4. Extra Reals complete (Have {existing_ex_real} >= Requested {args.n_extra_real}) ---")

    logger.info("Dataset curation complete. All mappings updated.")

if __name__ == "__main__":
    main()