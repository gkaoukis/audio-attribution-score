"""
Precompute features for ALL datasets and build a master metadata index.

This script iterates over every dataset defined in data_config.yaml,
extracts features (classical, ai_detection, fakeprint, and optionally
MERT/CLAP embeddings), saves .npz caches, and produces a unified
feature_metadata.csv mapping each file to its dataset, subset, labels,
and feature cache path.

Usage:
    # CPU-only (no embeddings)
    python precompute_all.py --skip_embeddings

    # With GPU embeddings
    python precompute_all.py

    # Force recompute
    python precompute_all.py --force --skip_embeddings
"""

import argparse
import hashlib
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16000
DATA_DIR = Path("data")
CACHE_DIR = Path("feature_cache")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_audio(path: Path, sr: int = TARGET_SR):
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def extract_cpu_features(audio_path: Path, cache_dir: Path, sr: int = TARGET_SR, force: bool = False):
    """Extract classical + ai_detection + fakeprint (CPU-only)."""
    from feature_extraction.classical import extract_classical_features, extract_classical_chunked, CLASSICAL_DIM
    from feature_extraction.ai_detection import extract_ai_detection_features, AI_DETECTION_DIM
    from feature_extraction.fakeprint import extract_fakeprint, FAKEPRINT_DIM

    fid = file_hash(audio_path)
    cache_path = cache_dir / f"{fid}.npz"

    if cache_path.exists() and not force:
        try:
            data = dict(np.load(cache_path, allow_pickle=True))
            if "classical" in data and "ai_detection" in data and "fakeprint" in data:
                return fid, cache_path
        except Exception:
            pass

    y = load_audio(audio_path, sr=sr)
    if y is None:
        return None, None

    result = {}
    try:
        result["classical"] = extract_classical_features(y, sr=sr)
    except Exception as e:
        logger.warning(f"Classical failed {audio_path}: {e}")
        result["classical"] = np.zeros(CLASSICAL_DIM, dtype=np.float32)

    try:
        result["classical_chunks"] = extract_classical_chunked(y, sr=sr)
    except Exception:
        result["classical_chunks"] = result["classical"].reshape(1, -1)

    try:
        result["ai_detection"] = extract_ai_detection_features(y, sr=sr)
    except Exception as e:
        logger.warning(f"AI detection failed {audio_path}: {e}")
        result["ai_detection"] = np.zeros(AI_DETECTION_DIM, dtype=np.float32)

    try:
        result["fakeprint"] = extract_fakeprint(y, sr=sr)
    except Exception as e:
        logger.warning(f"Fakeprint failed {audio_path}: {e}")
        result["fakeprint"] = np.zeros(FAKEPRINT_DIM, dtype=np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return fid, cache_path


def _worker(args):
    path, cache_dir, sr, force = args
    return extract_cpu_features(Path(path), cache_dir, sr, force)

# dataset discovery 

def discover_sonics(cfg):
    """Build file list with metadata for all SONICS subsets."""
    rows = []

    # Lyric pairs - fake
    pair_fake_meta = cfg["sonics"]["pair_fakes"]
    if (DATA_DIR / pair_fake_meta["metadata_csv"]).exists():
        df = pd.read_csv(DATA_DIR / pair_fake_meta["metadata_csv"], low_memory=False)
        audio_dir = DATA_DIR / pair_fake_meta["audio_dir"]
        for _, r in df.iterrows():
            fp = audio_dir / r["filename"]
            if not str(fp).endswith(".wav"):
                fp = fp.with_suffix(".wav")
            if fp.exists():
                rows.append({
                    "path": str(fp), "dataset": "sonics", "subset": "lyric_pair_fake",
                    "is_ai": True, "algorithm": r.get("algorithm", "unknown"),
                    "genre": r.get("genre", ""), "mood": r.get("mood", ""),
                    "topic": r.get("topic", ""), "filename": r["filename"],
                })

    # Lyric pairs - real (files named by youtube_id, not by 'filename' column)
    pair_real_meta = cfg["sonics"]["pair_reals"]
    if (DATA_DIR / pair_real_meta["metadata_csv"]).exists():
        df = pd.read_csv(DATA_DIR / pair_real_meta["metadata_csv"], low_memory=False)
        audio_dir = DATA_DIR / pair_real_meta["audio_dir"]
        for _, r in df.iterrows():
            # Real files are named by youtube_id
            yt_id = r.get("youtube_id", "")
            if pd.notna(yt_id) and yt_id:
                fp = audio_dir / f"{yt_id}.wav"
            else:
                fp = audio_dir / r["filename"]
                if not str(fp).endswith(".wav"):
                    fp = fp.with_suffix(".wav")
            if fp.exists():
                rows.append({
                    "path": str(fp), "dataset": "sonics", "subset": "lyric_pair_real",
                    "is_ai": False, "algorithm": "real",
                    "genre": "", "mood": "", "topic": "",
                    "filename": fp.name, "youtube_id": str(yt_id),
                })

    # Sibling pairs (AI vs AI same base)
    sib_meta = cfg["sonics"]["sibling_fakes"]
    if (DATA_DIR / sib_meta["metadata_csv"]).exists():
        df = pd.read_csv(DATA_DIR / sib_meta["metadata_csv"], low_memory=False)
        audio_dir = DATA_DIR / sib_meta["audio_dir"]
        for _, r in df.iterrows():
            fp = audio_dir / r["filename"]
            if not str(fp).endswith(".wav"):
                fp = fp.with_suffix(".wav")
            if fp.exists():
                rows.append({
                    "path": str(fp), "dataset": "sonics", "subset": "sibling_fake",
                    "is_ai": True, "algorithm": r.get("algorithm", "unknown"),
                    "genre": r.get("genre", ""), "mood": r.get("mood", ""),
                    "topic": r.get("topic", ""), "filename": r["filename"],
                })

    # Extra fakes
    extra_fake = cfg["sonics"].get("extra_fakes")
    if extra_fake and (DATA_DIR / extra_fake["metadata_csv"]).exists():
        df = pd.read_csv(DATA_DIR / extra_fake["metadata_csv"], low_memory=False)
        audio_dir = DATA_DIR / extra_fake["audio_dir"]
        for _, r in df.iterrows():
            fp = audio_dir / r["filename"]
            if not str(fp).endswith(".wav"):
                fp = fp.with_suffix(".wav")
            if fp.exists():
                rows.append({
                    "path": str(fp), "dataset": "sonics", "subset": "extra_fake",
                    "is_ai": True, "algorithm": r.get("algorithm", "unknown"),
                    "genre": r.get("genre", ""), "mood": r.get("mood", ""),
                    "topic": r.get("topic", ""), "filename": r["filename"],
                })

    # Extra reals (files named by youtube_id)
    extra_real = cfg["sonics"].get("extra_reals")
    if extra_real and (DATA_DIR / extra_real["metadata_csv"]).exists():
        df = pd.read_csv(DATA_DIR / extra_real["metadata_csv"], low_memory=False)
        audio_dir = DATA_DIR / extra_real["audio_dir"]
        for _, r in df.iterrows():
            yt_id = r.get("youtube_id", "")
            if pd.notna(yt_id) and yt_id:
                fp = audio_dir / f"{yt_id}.wav"
            else:
                fp = audio_dir / r["filename"]
                if not str(fp).endswith(".wav"):
                    fp = fp.with_suffix(".wav")
            if fp.exists():
                rows.append({
                    "path": str(fp), "dataset": "sonics", "subset": "extra_real",
                    "is_ai": False, "algorithm": "real",
                    "genre": "", "mood": "", "topic": "",
                    "filename": fp.name, "youtube_id": str(yt_id),
                })
    return rows

def discover_fakemusiccaps(cfg):
    rows = []
    fmc = cfg["fakemusiccaps"]
    meta_path = DATA_DIR / fmc["metadata_csv"]
    if not meta_path.exists():
        return rows
    df = pd.read_csv(meta_path, low_memory=False)
    audio_dir = DATA_DIR / fmc["audio_dir"]
    for _, r in df.iterrows():
        folder = r.get("folder", "")
        fname = r["filename"]
        # Files may be flat in audio_dir or in folder subdirectory
        fp = audio_dir / folder / fname
        if not fp.exists():
            fp = audio_dir / fname  # flat layout
        if fp.exists():
            # folder name is the generator label (e.g., "MusicGen_medium")
            rows.append({
                "path": str(fp), "dataset": "fakemusiccaps", "subset": folder,
                "is_ai": True, "algorithm": folder,
                "genre": "", "mood": "", "topic": "",
                "filename": fname, "caption": r.get("caption", ""),
            })
    return rows


def discover_smp(cfg):
    rows = []
    smp = cfg["smp_dataset"]
    meta_path = DATA_DIR / smp["metadata_csv"]
    if not meta_path.exists():
        return rows
    df = pd.read_csv(meta_path, low_memory=False)
    audio_dir = DATA_DIR / smp["audio_dir"]

    for _, r in df.iterrows():
        pair_dir = audio_dir / str(int(r["pair_number"]))
        if not pair_dir.exists():
            continue
        for wav in pair_dir.glob("*.wav"):
            # Determine if original or comparison
            fname_clean = wav.stem.lower()
            ori_clean = str(r.get("ori_title", "")).lower().replace(" ", "")
            comp_clean = str(r.get("comp_title", "")).lower().replace(" ", "")
            is_ori = ori_clean[:10] in fname_clean.replace(" ", "") if ori_clean else False
            rows.append({
                "path": str(wav), "dataset": "smp", "subset": r.get("relation", "unknown"),
                "is_ai": False,  # SMP is real plagiarism pairs
                "algorithm": "real",
                "genre": "", "mood": "", "topic": "",
                "filename": wav.name,
                "pair_number": int(r["pair_number"]),
                "relation": r.get("relation", ""),
                "is_original": is_ori,
            })
    return rows


def discover_echoes(cfg):
    rows = []
    echoes = cfg["echoes"]

    # Generated tracks
    gen_meta_path = DATA_DIR / echoes["generated"]["metadata_csv"]
    if gen_meta_path.exists():
        df = pd.read_csv(gen_meta_path, low_memory=False)
        # processed_path is relative to echoes_processed/, not the template audio_dir
        base_dir = DATA_DIR / "echoes_processed"
        for _, r in df.iterrows():
            pp = r.get("processed_path", "")
            if pd.isna(pp) or not pp:
                continue
            fp = base_dir / pp
            if fp.exists():
                rows.append({
                    "path": str(fp), "dataset": "echoes", "subset": "generated",
                    "is_ai": True, "algorithm": r.get("generator", "unknown"),
                    "genre": r.get("genre", ""), "mood": "", "topic": "",
                    "filename": fp.name,
                    "echoes_type": r.get("type", ""),
                    "description": r.get("description", ""),
                    "original_audio": r.get("original_audio", ""),
                })

    # Originals
    ori_meta_path = DATA_DIR / echoes["originals"]["metadata_csv"]
    if ori_meta_path.exists():
        df = pd.read_csv(ori_meta_path, low_memory=False)
        base_dir = DATA_DIR / "echoes_processed"
        for _, r in df.iterrows():
            pp = r.get("processed_path", "")
            if pd.isna(pp) or not pp:
                continue
            fp = base_dir / pp
            if fp.exists():
                rows.append({
                    "path": str(fp), "dataset": "echoes", "subset": "original",
                    "is_ai": False, "algorithm": "real",
                    "genre": "", "mood": "", "topic": "",
                    "filename": fp.name,
                    "original_audio": r.get("original_audio", ""),
                })
    return rows



def main():
    parser = argparse.ArgumentParser(description="Precompute features for all datasets")
    parser.add_argument("--mode", type=str, choices=["all", "cpu", "gpu"], default="all",
                        help="Extraction mode: 'all' (CPU+GPU), 'cpu' (no embeddings), 'gpu' (only embeddings)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if cache exists")
    parser.add_argument("--max_workers", type=int, default=24,
                        help="Parallel CPU workers")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Output directory. Defaults to 'feature_cache_{mode}'")
    args = parser.parse_args()

    # All discover_* helpers use the module-level DATA_DIR; update it here.
    global DATA_DIR
    script_dir = Path(__file__).resolve().parent
    DATA_DIR = (script_dir / "data").resolve()

    if args.cache_dir is None:
        cache_dir = (script_dir / f"feature_cache_{args.mode}").resolve()
    else:
        cache_dir = Path(args.cache_dir).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory:  {DATA_DIR}")
    logger.info(f"Cache directory: {cache_dir}")

    with open(script_dir / "data_config.yaml") as f:
        cfg = yaml.safe_load(f)["datasets"]

    # Discover all files
    logger.info("Discovering audio files across all datasets...")
    all_rows = []
    all_rows.extend(discover_sonics(cfg))
    all_rows.extend(discover_fakemusiccaps(cfg))
    all_rows.extend(discover_smp(cfg))
    all_rows.extend(discover_echoes(cfg))

    logger.info(f"Found {len(all_rows)} audio files total")

    file_ids = [None] * len(all_rows)
    cache_paths = [None] * len(all_rows)

    # ── Phase 1: CPU features (classical + ai_detection + fakeprint) ──
    if args.mode in ["all", "cpu"]:
        logger.info(f"Phase 1: Extracting CPU features using {args.max_workers} workers...")
        worker_args = [(r["path"], cache_dir, TARGET_SR, args.force) for r in all_rows]

        if args.max_workers > 1:
            with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
                for i, (fid, cp) in enumerate(tqdm(
                    pool.map(_worker, worker_args), total=len(all_rows), desc="CPU features"
                )):
                    file_ids[i] = fid
                    cache_paths[i] = cp
        else:
            for i, wa in enumerate(tqdm(worker_args, desc="CPU features")):
                fid, cp = _worker(wa)
                file_ids[i] = fid
                cache_paths[i] = cp
    else:
        # If skipping Phase 1, we still need to calculate file hashes for Phase 2
        logger.info("Skipping Phase 1 (CPU features) because mode='gpu'...")
        for i, row in enumerate(all_rows):
            fid = file_hash(Path(row["path"]))
            file_ids[i] = fid
            cache_paths[i] = cache_dir / f"{fid}.npz"

    # ── Phase 2: GPU embeddings (MERT + CLAP) ──
    if args.mode in ["all", "gpu"]:
        logger.info("Phase 2: Extracting MERT + CLAP embeddings sequentially...")
        from feature_extraction.embeddings import extract_mert_embedding, extract_clap_embedding, unload_models

        for i, row in enumerate(tqdm(all_rows, desc="GPU embeddings")):
            fid = file_ids[i]
            cp = cache_paths[i]
            
            # Load existing data if appending to a file, otherwise start fresh
            data = {}
            if cp and cp.exists():
                try:
                    data = dict(np.load(cp, allow_pickle=True))
                except Exception:
                    pass

            if "mert" in data and "clap" in data and not args.force:
                continue

            y = load_audio(Path(row["path"]), sr=TARGET_SR)
            if y is None:
                continue

            try:
                data["mert"] = extract_mert_embedding(y, sr=TARGET_SR)
            except Exception as e:
                logger.warning(f"MERT failed {row['path']}: {e}")
                data["mert"] = np.zeros(1024, dtype=np.float32)

            try:
                data["clap"] = extract_clap_embedding(y, sr=TARGET_SR)
            except Exception as e:
                logger.warning(f"CLAP failed {row['path']}: {e}")
                data["clap"] = np.zeros(512, dtype=np.float32)

            save_dict = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(cp, **save_dict)

        unload_models()

    logger.info("Building feature metadata index...")
    meta_rows = []
    for i, row in enumerate(all_rows):
        if file_ids[i] is None:
            continue
        meta_rows.append({
            "file_id": file_ids[i],
            "cache_path": str(cache_paths[i]),
            **{k: v for k, v in row.items() if k != "path"},
            "audio_path": row["path"],
        })

    df_meta = pd.DataFrame(meta_rows)
    meta_path = cache_dir / "feature_metadata.csv"
    df_meta.to_csv(meta_path, index=False)
    logger.info(f"Saved metadata for {len(df_meta)} files to {meta_path}")

if __name__ == "__main__":
    main()