"""Download FakeMusicCaps dataset from Zenodo.

Usage:
    python fetch_fakemusiccaps.py
    python fetch_fakemusiccaps.py --n_samples_per_folder 400
"""
import argparse
import hashlib
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import pandas as pd
import requests
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000
ZIP_URL = "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip?download=1"
ZIP_MD5 = "db418dc95ab7dc378a55f29d6021fd66"
MUSICCAPS_CSV_URL = "https://huggingface.co/datasets/google/MusicCaps/resolve/main/musiccaps-public.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Download FakeMusicCaps from Zenodo.")
    p.add_argument("--n_samples_per_folder", type=int, default=400, help="Tracks to extract per folder (e.g. 400 * 5 folders = 2000 total)")
    p.add_argument("--output_dir", type=Path, default=Path("fakemusiccaps"), help="Where to save audio and metadata")
    return p.parse_args()


# ── Download helpers ──────────────────────────────────────────────────

def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_zip(dest: Path):
    if dest.exists():
        logger.info("Checking md5 of existing zip...")
        if md5_file(dest) == ZIP_MD5:
            logger.info("Zip already downloaded and verified, skipping.")
            return
        logger.warning("Zip exists but md5 mismatch - re-downloading.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading FakeMusicCaps.zip...")
    resp = requests.get(ZIP_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            pbar.update(len(chunk))

    actual = md5_file(dest)
    if actual != ZIP_MD5:
        logger.error(f"MD5 mismatch after download: expected {ZIP_MD5}, got {actual}")
        raise SystemExit(1)
    logger.info("Download complete, md5 verified.")


def fetch_musiccaps_captions(cache_dir: Path) -> dict[str, str]:
    csv_path = cache_dir / "musiccaps-public.csv"
    if not csv_path.exists():
        logger.info("Downloading MusicCaps CSV...")
        try:
            resp = requests.get(MUSICCAPS_CSV_URL, timeout=60)
            resp.raise_for_status()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_bytes(resp.content)
        except requests.RequestException as e:
            logger.warning(f"Failed to download MusicCaps CSV: {e}. Captions will be empty.")
            return {}

    df = pd.read_csv(csv_path)
    if "ytid" not in df.columns or "caption" not in df.columns:
        return {}
    return {
        str(row["ytid"]).strip(): str(row["caption"]).strip()
        for _, row in df.iterrows()
        if pd.notna(row["caption"])
    }


# ── Audio conversion ─────────────────────────────────────────────────

def to_16k_mono(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    try:
        y, _ = librosa.load(str(src), sr=TARGET_SR, mono=True)
        sf.write(str(dst), y, TARGET_SR, subtype="PCM_16")
        return True
    except Exception as e:
        logger.warning(f"Conversion failed for {src.name}: {e}")
        return False


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = args.output_dir
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metadata.csv"
    staging_dir = output_dir / "_staging"

    # Load existing metadata to skip already-processed files
    existing = set()
    if csv_path.exists():
        existing = set(pd.read_csv(csv_path)["filename"].tolist())
        logger.info(f"Found {len(existing)} already-processed files in metadata.csv.")

    # Download and verify zip
    zip_path = staging_dir / "FakeMusicCaps.zip"
    download_zip(zip_path)

    # Extract
    extract_dir = staging_dir / "extracted"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        extract_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        logger.info("Zip already extracted, skipping extraction.")

    # Find all folders with WAVs, skip __MACOSX
    folders = sorted(
        d for d in extract_dir.rglob("*")
        if d.is_dir()
        and "__MACOSX" not in d.parts
        and any(d.glob("*.wav"))
    )
    if not folders:
        logger.error("No folders with WAV files found after extraction.")
        return
    logger.info(f"Found {len(folders)} folders: {[f.name for f in folders]}")

    # Fetch captions
    captions = fetch_musiccaps_captions(staging_dir)

    # Build tasks based on the per-folder limit
    per_folder = args.n_samples_per_folder
    tasks = []

    for folder in folders:
        wavs = sorted(f for f in folder.glob("*.wav") if not f.name.startswith("._"))
        count = 0
        for wav in wavs:
            if count >= per_folder:
                break
                
            out_name = f"{folder.name}_{wav.stem}.wav"
            
            # If we already have it, skip the heavy work but keep counting
            if out_name in existing:
                count += 1
                continue
                
            tasks.append((wav, audio_dir / out_name, {
                "filename": out_name,
                "folder": folder.name,
                "caption": captions.get(wav.stem, ""),
            }))
            count += 1

    new_metadata = []

    if tasks:
        logger.info(f"Converting {len(tasks)} NEW audio files to 16kHz PCM_16...")

        def _worker(t):
            src, dst, meta = t
            return meta if to_16k_mono(src, dst) else None

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_worker, t) for t in tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                result = f.result()
                if result:
                    new_metadata.append(result)
    else:
        logger.info(f"No new files to process (already reached {per_folder} per folder).")

    # Append ONLY new metadata to the CSV
    if new_metadata:
        df = pd.DataFrame(new_metadata)
        # mode='a' appends data. header is added only if the file doesn't exist yet.
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        logger.info(f"Appended {len(df)} new records to metadata.csv.")
        
    # Verify final count
    total_records = len(pd.read_csv(csv_path)) if csv_path.exists() else 0
    logger.info(f"Done - {total_records} total samples now tracked in metadata.")

if __name__ == "__main__":
    main()