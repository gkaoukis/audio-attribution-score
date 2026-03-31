"""Fetch and process the SMP dataset.

Usage (run from data/):
    python fetch_smp.py
    python fetch_smp.py --max_workers 3 --skip_download
"""
import argparse
import logging
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/Mippia/smp_dataset.git"
TARGET_SR = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch and process SMP dataset.")
    parser.add_argument("--data_dir", type=str, default=".", help="Root data directory (default: current dir)")
    parser.add_argument("--max_workers", type=int, default=3, help="Parallel download workers")
    parser.add_argument("--skip_download", action="store_true", help="Skip cloning & downloading, just validate & convert")
    return parser.parse_args()


# ── Clone ─────────────────────────────────────────────────────────────

def clone_repo(data_dir: Path):
    repo_dir = data_dir / "smp_dataset"
    if repo_dir.exists():
        logger.info("smp_dataset/ already exists — skipping clone.")
        return repo_dir
    logger.info(f"Cloning {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
    return repo_dir


# ── Download ──────────────────────────────────────────────────────────

def download_pair(row, output_dir: Path):
    """Download original and comparison audio for one pair."""
    pair_dir = output_dir / str(row["pair_number"])
    pair_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for title_col, link_col in [("ori_title", "ori_link"), ("comp_title", "comp_link")]:
        title = str(row[title_col])
        link = str(row[link_col]).strip()
        clean = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
        out_file = pair_dir / f"{clean}.wav"

        if out_file.exists():
            results.append((True, out_file))
            continue

        try:
            subprocess.run([
                "yt-dlp", "-x",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "-o", str(out_file),
                link,
            ], check=True, capture_output=True)
            results.append((True, out_file))
            time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to download '{title}': {e}")
            results.append((False, out_file))

    return results


def download_all(csv_path: Path, output_dir: Path, max_workers: int):
    """Download all pairs from the CSV."""
    output_dir.mkdir(exist_ok=True)
    df = pd.read_csv(csv_path)
    unique = df.drop_duplicates(subset=["pair_number", "ori_title", "comp_title", "ori_link", "comp_link"])
    logger.info(f"Downloading {len(unique)} pairs ({len(unique) * 2} songs)...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_pair, row, output_dir): row["pair_number"] for _, row in unique.iterrows()}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Downloading pairs"):
            f.result()


# ── Validate & Quarantine ────────────────────────────────────────────

def validate_and_quarantine(csv_path: Path, dataset_dir: Path, output_16k: Path):
    """Check pairs for completeness. Move incomplete pairs to orphan/ in both dirs."""
    df = pd.read_csv(csv_path)
    unique = df.drop_duplicates(subset=["pair_number", "ori_title", "comp_title"])

    orphan_src = dataset_dir / "orphan"
    orphan_16k = output_16k / "orphan"

    complete_pairs = 0
    incomplete_pairs = 0

    for _, row in unique.iterrows():
        pair_num = str(row["pair_number"])
        pair_dir = dataset_dir / pair_num

        # Build expected filenames for this pair
        expected = []
        for title_col in ["ori_title", "comp_title"]:
            title = str(row[title_col])
            clean = "".join(c if c.isalnum() or c in " -_" else "_" for c in title).strip()
            expected.append(f"{clean}.wav")

        # Check if both files exist
        all_present = pair_dir.exists() and all((pair_dir / f).exists() for f in expected)

        if all_present:
            complete_pairs += 1
        else:
            incomplete_pairs += 1
            missing = [f for f in expected if not pair_dir.exists() or not (pair_dir / f).exists()]

            # Move to orphan/ in final_dataset
            if pair_dir.exists():
                dst = orphan_src / pair_num
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pair_dir), str(dst))

            # Move to orphan/ in smp_dataset_16
            pair_16k = output_16k / pair_num
            if pair_16k.exists():
                dst = orphan_16k / pair_num
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pair_16k), str(dst))

            logger.warning(f"Pair {pair_num} incomplete → orphan/ (missing: {missing})")

    logger.info(f"Complete pairs: {complete_pairs}")
    if incomplete_pairs:
        logger.warning(f"Incomplete pairs moved to orphan/: {incomplete_pairs}")

    return complete_pairs, incomplete_pairs


# ── Convert to 16kHz ─────────────────────────────────────────────────

def convert_dataset(src_dir: Path, dst_dir: Path):
    """Mirror src_dir pair folders into dst_dir with 16kHz mono PCM_16 WAVs.
    
    Skips orphan/ directory."""
    wavs = [
        w for w in src_dir.rglob("*.wav")
        if w.relative_to(src_dir).parts[0] != "orphan"
    ]

    if not wavs:
        logger.warning(f"No WAV files found in {src_dir}")
        return

    logger.info(f"Converting {len(wavs)} files to 16kHz mono PCM_16 → {dst_dir}")
    skipped = 0

    for wav in tqdm(wavs, desc="Converting"):
        rel = wav.relative_to(src_dir)
        out = dst_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)

        if out.exists():
            skipped += 1
            continue

        try:
            y, _ = librosa.load(str(wav), sr=TARGET_SR, mono=True)
            sf.write(str(out), y, TARGET_SR, subtype="PCM_16")
        except Exception as e:
            logger.warning(f"Conversion failed for {rel}: {e}")

    if skipped:
        logger.info(f"Skipped {skipped} already-converted files.")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    repo_dir = data_dir / "smp_dataset"
    csv_path = repo_dir / "Final_dataset_pairs.csv"
    dataset_dir = repo_dir / "final_dataset"
    output_16k = data_dir / "smp_dataset_16"

    # Clone
    if not args.skip_download:
        clone_repo(data_dir)

    if not csv_path.exists():
        logger.error(f"CSV not found at {csv_path}. Check the repo.")
        return

    # Download
    if not args.skip_download:
        download_all(csv_path, dataset_dir, args.max_workers)

    # Convert first (so quarantine can move from both dirs)
    convert_dataset(dataset_dir, output_16k)

    # Validate & quarantine incomplete pairs
    logger.info("Validating pairs...")
    validate_and_quarantine(csv_path, dataset_dir, output_16k)

    logger.info("Done.")


if __name__ == "__main__":
    main()