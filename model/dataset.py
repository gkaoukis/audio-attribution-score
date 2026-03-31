"""
Pair dataset construction for training the attribution model.

Training split:
    Positive pairs:
        - SONICS lyric pairs: real ↔ AI fake (same lyrics), is_attribution=1
        - SONICS sibling pairs: AI ↔ AI from same lyrics/style, similarity=0.8
        - SMP pairs: original ↔ cover/plagiarism, is_attribution=1
    Negative pairs:
        - SONICS fake negatives: same-generator and cross-generator (different content)
        - Fake vs unrelated real (SONICS fakes ↔ extra real tracks)
        - FakeMusicCaps cross-generator, cross-topic negatives

Validation split:
    - Held-out portion of the above (strict track-level splitting)
    - Echoes: ATA (audio-to-audio) pairs only (original FMA ↔ AI derivative)

AI head:
    Trained through all pairs — each track carries an is_ai label.
"""

import hashlib
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_features(cache_path: str, max_chunks: int = 12) -> dict:
    """Load precomputed features from .npz cache file.

    If the track has more than max_chunks windows, randomly sample max_chunks
    of them (without replacement) and sort by index to preserve temporal order.
    Random sampling provides data augmentation during training and prevents
    positional bias from head-truncation.
    """
    data = dict(np.load(cache_path, allow_pickle=True))
    if "classical_chunks" in data:
        chunks = data["classical_chunks"]
        n = chunks.shape[0]
        if n > max_chunks:
            indices = np.sort(np.random.choice(n, size=max_chunks, replace=False))
            data["classical_chunks"] = chunks[indices]
    return data



def _parse_sonics_filename(filename: str) -> tuple:
    """Parse SONICS filename into (group_id, generator, variant).

    e.g., 'fake_40011_suno_0.wav' -> ('40011', 'suno', '0')
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "fake":
        return parts[1], parts[2], parts[3]
    return stem, "unknown", "0"


# ── Feature set filtering ─────────────────────────────────────────────


FEATURE_SETS = {
    "basic": {
        "chunk_keys": ["classical_chunks"],
        "track_keys": ["classical"],
        "ai_keys": ["ai_detection", "fakeprint"],
        "emb_keys": [],
    },
    "embedding": {
        "chunk_keys": [],  
        "track_keys": [],
        "ai_keys": [],
        "emb_keys": ["mert", "clap"],
    },
    "advanced": {
        "chunk_keys": ["classical_chunks"],
        "track_keys": ["classical"],
        "ai_keys": ["ai_detection", "fakeprint"],
        "emb_keys": ["mert", "clap"],
    },
    "mix": {
        "chunk_keys": ["classical_chunks"],
        "track_keys": ["classical"],
        "ai_keys": ["ai_detection", "fakeprint"],
        "emb_keys": ["mert", "clap"],
        # Lyrics are handled dynamically in network.py
    },
}


# ── Training dataset ──────────────────────────────────────────────────


class PairDataset(Dataset):
    """Training/Validation dataset of audio pairs with strict track-level splitting."""

    def __init__(
        self,
        data_dir: str = "data",
        cache_dir: str = "feature_cache",
        neg_ratio: float = 1.5,
        max_chunks: int = 12,
        seed: int = 42,
        split: str = "train",
        val_ratio: float = 0.15,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.max_chunks = max_chunks
        self.neg_ratio = neg_ratio
        self.split = split
        self.val_ratio = val_ratio
        self.rng = random.Random(seed)

        # Build file_hash -> cache_path mapping from the feature metadata CSV.
        self.path_to_cache = {}
        self.smp_cache_groups = defaultdict(list)
        
        meta_file = self.cache_dir / "feature_metadata.csv"
        if meta_file.exists():
            df_meta = pd.read_csv(meta_file)
            for _, row in df_meta.iterrows():
                # Store the full normalized path string as the key
                audio_path = Path(str(row['audio_path']).replace("\\", "/")).as_posix()
                cache_path = row['cache_path']
                self.path_to_cache[audio_path] = cache_path
                
                if "smp_dataset_16" in audio_path:
                    try:
                        parts = Path(audio_path).parts
                        idx = parts.index("smp_dataset_16")
                        pair_num = int(parts[idx + 1])
                        self.smp_cache_groups[pair_num].append(cache_path)
                    except (ValueError, IndexError):
                        pass
        else:
            logger.warning(f"No metadata found at {meta_file}.")
            self.path_to_cache = {npz.stem + ".wav": str(npz) for npz in self.cache_dir.glob("*.npz")}
        self.pairs = []
        self._used_pairs = set()

        # Pools for negative mining (populated during positive pair building)
        self._fake_by_gen = defaultdict(list)  # generator -> [cache_paths]
        self._cache_to_group = {}  # cache_path -> group_id
        self._cache_to_gen = {}  # cache_path -> generator
        self._sibling_groups = defaultdict(set)  # group_id -> {cache_paths}
        self._real_pool = []
        self._group_metadata = {}  # group_id -> {genre, topic, algorithm}
        self._fmc_pool = []  # FMC cache paths for cross-dataset pairing

        self._load_group_metadata()
        logger.info(f"Building {split.upper()} dataset...")

        # Pre-load FMC tracks once; shared by positives and negatives below.
        self._build_fmc_tracks()

        # === Positives ===
        self._build_sonics_lyric_pairs()
        self._build_sonics_sibling_pairs()
        self._build_smp_pairs()
        self._build_fmc_same_prompt_pairs()

        # === Negatives ===
        self._build_sonics_fake_negatives()
        self._build_fake_vs_real_negatives()
        self._build_fmc_negatives()
        self._build_cross_dataset_negatives()

        self._log_pair_stats()
        self.rng.shuffle(self.pairs)
        logger.info(f"Total {split.upper()} pairs: {len(self.pairs)}")

    def _resolve_cache(self, audio_path: Path) -> Optional[str]:
        search_path = audio_path.as_posix()
        # Find the cache path where the end of the stored path matches our search
        for k, v in self.path_to_cache.items():
            if k.endswith(search_path) or search_path.endswith(k):
                if Path(v).exists():
                    return v
        return None

    def _get_split_items(self, items) -> list:
        items = list(items)
        self.rng.shuffle(items)
        cutoff = int(len(items) * (1.0 - self.val_ratio))
        return items[:cutoff] if self.split == "train" else items[cutoff:]

    def _register_pair(self, cache_a: str, cache_b: str) -> bool:
        """Register a pair to avoid duplicates. Returns True if new."""
        key = tuple(sorted([cache_a, cache_b]))
        if key in self._used_pairs:
            return False
        self._used_pairs.add(key)
        return True

    def _register_fake(self, cache: str, group_id: str, generator: str):
        """Register a fake track in the pools for later negative mining."""
        if cache not in self._cache_to_group:
            self._cache_to_group[cache] = group_id
            self._cache_to_gen[cache] = generator
            self._fake_by_gen[generator].append(cache)

    def _load_group_metadata(self):
        """Load genre/topic/algorithm per group_id from extra_fake_metadata."""
        csv = self.data_dir / "sonics" / "extra_fake_metadata.csv"
        if not csv.exists():
            return
        df = pd.read_csv(csv, on_bad_lines="skip")
        for _, row in df.iterrows():
            gid = str(row.get("id", ""))
            if not gid or pd.isna(gid):
                continue
            self._group_metadata[gid] = {
                "genre": str(row.get("genre", "")).lower().strip(),
                "topic": str(row.get("topic", "")).lower().strip(),
                "algorithm": str(row.get("algorithm", "")).lower().strip(),
            }
        logger.info(f"  Loaded metadata for {len(self._group_metadata)} groups")

    def _negative_similarity(self, cache_a: str, cache_b: str) -> float:
        """Compute soft similarity for same-generator negatives using metadata overlap."""
        group_a = self._cache_to_group.get(cache_a, "")
        group_b = self._cache_to_group.get(cache_b, "")
        meta_a = self._group_metadata.get(group_a, {})
        meta_b = self._group_metadata.get(group_b, {})

        genre_a, genre_b = meta_a.get("genre", ""), meta_b.get("genre", "")
        topic_a, topic_b = meta_a.get("topic", ""), meta_b.get("topic", "")

        if genre_a and genre_a == genre_b:
            return 0.15
        if topic_a and topic_a == topic_b:
            return 0.1
        return 0.05

    # ── 1. SONICS Lyric Pairs: Real ↔ AI Fake (same lyrics) ──

    def _build_sonics_lyric_pairs(self):
        csv_path = self.data_dir / "sonics" / "lyric_pairs_mapping.csv"
        if not csv_path.exists():
            logger.warning("lyric_pairs_mapping.csv not found, skipping lyric pairs")
            return

        df = pd.read_csv(csv_path)
        fake_dir = self.data_dir / "sonics" / "pairs_fake_16"
        real_dir = self.data_dir / "sonics" / "pairs_real_16"

        split_indices = set(self._get_split_items(range(len(df))))

        count = 0
        for idx, row in df.iterrows():
            if idx not in split_indices:
                continue

            fake_name = row["fake_filename"]
            if not fake_name.endswith(".wav"):
                fake_name += ".wav"
            real_id = row["real_youtube_id"]

            cache_fake = self._resolve_cache(fake_dir / fake_name)
            cache_real = self._resolve_cache(real_dir / f"{real_id}.wav")

            if cache_fake and cache_real and self._register_pair(cache_fake, cache_real):
                group_id, gen, _ = _parse_sonics_filename(fake_name)
                self.pairs.append({
                    "cache_a": cache_real,
                    "cache_b": cache_fake,
                    "similarity": 1.0,
                    "is_ai_a": 0.0,
                    "is_ai_b": 1.0,
                    "is_attribution": 1.0,
                    "source": "sonics_lyric_pos",
                })
                self._register_fake(cache_fake, group_id, gen)
                self._real_pool.append(cache_real)
                count += 1

        logger.info(f"  SONICS lyric pairs: {count}")

    # ── 2. SONICS Siblings: AI ↔ AI (same style/lyrics) ──

    def _build_sonics_sibling_pairs(self):
        csv_path = self.data_dir / "sonics" / "sibling_pairs_mapping.csv"
        if not csv_path.exists():
            logger.warning("sibling_pairs_mapping.csv not found, skipping sibling pairs")
            return

        df = pd.read_csv(csv_path)
        audio_dir = self.data_dir / "sonics" / "extra_fake_16"

        # Split by group_id (numeric prefix) to prevent track leakage
        group_ids = sorted(set(
            _parse_sonics_filename(row["sibling_0"])[0]
            for _, row in df.iterrows()
        ))
        split_groups = set(self._get_split_items(group_ids))

        count = 0
        for _, row in df.iterrows():
            sib0, sib1 = row["sibling_0"], row["sibling_1"]
            if not sib0.endswith(".wav"):
                sib0 += ".wav"
            if not sib1.endswith(".wav"):
                sib1 += ".wav"

            group_id = _parse_sonics_filename(sib0)[0]
            if group_id not in split_groups:
                continue

            cache_a = self._resolve_cache(audio_dir / sib0)
            cache_b = self._resolve_cache(audio_dir / sib1)

            if cache_a and cache_b and self._register_pair(cache_a, cache_b):
                gen_a = _parse_sonics_filename(sib0)[1]
                gen_b = _parse_sonics_filename(sib1)[1]

                self.pairs.append({
                    "cache_a": cache_a,
                    "cache_b": cache_b,
                    "similarity": 0.8,
                    "is_ai_a": 1.0,
                    "is_ai_b": 1.0,
                    "is_attribution": 0.0,
                    "source": "sonics_sibling_pos",
                })
                self._register_fake(cache_a, group_id, gen_a)
                self._register_fake(cache_b, group_id, gen_b)
                self._sibling_groups[group_id].update([cache_a, cache_b])
                count += 1

        logger.info(f"  SONICS sibling pairs: {count}")

    # ── 3. SMP: Covers/Plagiarism (Real ↔ Real) ──

    def _build_smp_pairs(self):
        csv_path = self.data_dir / "smp_dataset_16" / "Final_dataset_pairs.csv"
        if not csv_path.exists():
            logger.warning("SMP CSV not found, skipping SMP pairs")
            return

        df = pd.read_csv(csv_path)

        # Split by unique pair_number to prevent track leakage
        unique_pairs = sorted(df["pair_number"].unique().tolist())
        split_pairs = set(self._get_split_items(unique_pairs))

        count = 0
        for _, row in df[df["pair_number"].isin(split_pairs)].iterrows():
            pair_num = int(row["pair_number"])
            
            # Look up the caches directly from our metadata grouping, no globbing needed!
            caches = self.smp_cache_groups.get(pair_num, [])
            
            if len(caches) < 2:
                continue

            cache_a = caches[0]
            cache_b = caches[1]

            # Make sure the cache files actually exist on this machine
            if Path(cache_a).exists() and Path(cache_b).exists() and self._register_pair(cache_a, cache_b):
                self.pairs.append({
                    "cache_a": cache_a,
                    "cache_b": cache_b,
                    "similarity": 1.0,
                    "is_ai_a": 0.0,
                    "is_ai_b": 0.0,
                    "is_attribution": 1.0,
                    "source": "smp_pos",
                })
                self._real_pool.extend([cache_a, cache_b])
                count += 1

        logger.info(f"  SMP pairs: {count}")

    # ── 4. SONICS Fake Negatives (same-gen + cross-gen) ──

    def _build_sonics_fake_negatives(self):
        """Negative AI↔AI pairs: same generator (hard neg) and cross generator."""
        self._load_extra_fakes()

        # Deduplicate pools
        for gen in self._fake_by_gen:
            self._fake_by_gen[gen] = list(set(self._fake_by_gen[gen]))

        gens = [g for g in self._fake_by_gen if len(self._fake_by_gen[g]) >= 2]
        if not gens:
            return

        total_fakes = sum(len(self._fake_by_gen[g]) for g in gens)
        n_target = int(total_fakes * self.neg_ratio * 0.3)
        n_target = max(n_target, 50)
        half = n_target // 2

        # Same-generator negatives (hard negatives: same gen, different content)
        same_gen_count = 0
        for gen in gens:
            pool = list(self._fake_by_gen[gen])
            self.rng.shuffle(pool)
            for i in range(0, len(pool) - 1, 2):
                a, b = pool[i], pool[i + 1]
                group_a = self._cache_to_group.get(a)
                group_b = self._cache_to_group.get(b)
                if group_a and group_b and group_a == group_b:
                    continue  # same content group, skip
                if self._register_pair(a, b):
                    self.pairs.append({
                        "cache_a": a,
                        "cache_b": b,
                        "similarity": self._negative_similarity(a, b),
                        "is_ai_a": 1.0,
                        "is_ai_b": 1.0,
                        "is_attribution": 0.0,
                        "source": "sonics_same_gen_neg",
                    })
                    same_gen_count += 1
                if same_gen_count >= half:
                    break
            if same_gen_count >= half:
                break

        # Cross-generator negatives
        cross_gen_count = 0
        if len(gens) >= 2:
            gen_pairs = [(g1, g2) for g1 in gens for g2 in gens if g1 < g2]
            for g1, g2 in gen_pairs:
                pool1 = list(self._fake_by_gen[g1])
                pool2 = list(self._fake_by_gen[g2])
                self.rng.shuffle(pool1)
                self.rng.shuffle(pool2)
                for a, b in zip(pool1, pool2):
                    if self._register_pair(a, b):
                        self.pairs.append({
                            "cache_a": a,
                            "cache_b": b,
                            "similarity": 0.0,
                            "is_ai_a": 1.0,
                            "is_ai_b": 1.0,
                            "is_attribution": 0.0,
                            "source": "sonics_cross_gen_neg",
                        })
                        cross_gen_count += 1
                    if cross_gen_count >= half:
                        break
                if cross_gen_count >= half:
                    break

        logger.info(
            f"  SONICS fake negatives: {same_gen_count} same-gen, "
            f"{cross_gen_count} cross-gen"
        )

    def _load_extra_fakes(self):
        """Load non-sibling extra fakes into the pool for negative mining."""
        csv_path = self.data_dir / "sonics" / "extra_fake_metadata.csv"
        audio_dir = self.data_dir / "sonics" / "extra_fake_16"
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        split_indices = set(self._get_split_items(range(len(df))))

        registered = set(self._cache_to_group.keys())
        count = 0
        for idx, row in df.iterrows():
            if idx not in split_indices:
                continue
            fname = row.get("filename", "")
            if pd.isna(fname) or not fname:
                continue
            if not fname.endswith(".wav"):
                fname += ".wav"

            fp = audio_dir / fname
            cache = self._resolve_cache(fp)
            if cache and cache not in registered:
                gid, gen, _ = _parse_sonics_filename(fname)
                self._register_fake(cache, gid, gen)
                registered.add(cache)
                count += 1

        logger.info(f"  Loaded {count} extra fakes for negative mining")

    # ── 5. Fake vs Unrelated Real ──

    def _build_fake_vs_real_negatives(self):
        """Pair unrelated AI fakes with real tracks as negatives."""
        self._load_extra_reals()

        all_fakes = [c for caches in self._fake_by_gen.values() for c in caches]
        reals = list(set(self._real_pool))
        if not all_fakes or not reals:
            return

        self.rng.shuffle(all_fakes)
        self.rng.shuffle(reals)

        n_target = min(len(all_fakes), len(reals))
        n_target = max(n_target, 50)

        count = 0
        real_idx = 0
        for fake in all_fakes:
            if count >= n_target:
                break
            real = reals[real_idx % len(reals)]
            real_idx += 1
            if self._register_pair(fake, real):
                self.pairs.append({
                    "cache_a": real,
                    "cache_b": fake,
                    "similarity": 0.0,
                    "is_ai_a": 0.0,
                    "is_ai_b": 1.0,
                    "is_attribution": 0.0,
                    "source": "fake_vs_real_neg",
                })
                count += 1

        logger.info(f"  Fake vs real negatives: {count}")

    def _load_extra_reals(self):
        """Load extra real tracks into the real pool."""
        csv_path = self.data_dir / "sonics" / "extra_real_metadata.csv"
        real_dir = self.data_dir / "sonics" / "extra_real_16"
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)
        split_indices = set(self._get_split_items(range(len(df))))

        registered = set(self._real_pool)
        count = 0
        for idx, row in df.iterrows():
            if idx not in split_indices:
                continue
            yt_id = row.get("youtube_id")
            if pd.isna(yt_id):
                continue
            fp = real_dir / f"{yt_id}.wav"
            cache = self._resolve_cache(fp)
            if cache and cache not in registered:
                self._real_pool.append(cache)
                registered.add(cache)
                count += 1

        logger.info(f"  Loaded {count} extra reals for negative mining")

    # ── 6. FakeMusicCaps: shared loader + positives + negatives ──

    def _build_fmc_tracks(self):
        """Load and split all FMC tracks once; results stored on self.

        Sets:
            _fmc_by_yt:   yt_id  -> {generator: cache_path}  (split-filtered)
            _fmc_gen_tracks: generator -> [(cache_path, yt_id)]
            _fmc_pool:    flat list of all resolved cache paths
        """
        self._fmc_by_yt = {}
        self._fmc_gen_tracks = defaultdict(list)
        self._fmc_pool = []

        csv_path = self.data_dir / "fakemusiccaps" / "metadata.csv"
        audio_dir = self.data_dir / "fakemusiccaps" / "audio"
        if not csv_path.exists():
            return

        df = pd.read_csv(csv_path)

        # First pass: resolve all cache paths and group by yt_id.
        all_by_yt: dict = defaultdict(dict)
        for _, row in df.iterrows():
            gen = str(row.get("folder", "")).strip()
            fname = str(row["filename"])

            # Try folder-based layout first (data/fakemusiccaps/audio/{gen}/{fname}),
            # then fall back to flat layout (data/fakemusiccaps/audio/{fname}).
            # The path stored in the metadata CSV may use either layout, and
            # _resolve_cache's suffix-matching cannot bridge a missing directory
            # component, so we must construct the right path explicitly.
            cache = self._resolve_cache(audio_dir / gen / fname) if gen else None
            if not cache:
                cache = self._resolve_cache(audio_dir / fname)
            if not cache:
                continue

            # Extract the YouTube ID: filenames follow "{gen}_{yt_id}.wav".
            # If the pattern doesn't match, fall back to the full stem.
            if gen and fname.startswith(gen + "_"):
                yt_id = fname[len(gen) + 1:].replace(".wav", "")
            else:
                yt_id = fname.replace(".wav", "")

            all_by_yt[yt_id][gen] = cache

        # Split by YouTube ID for strict track-level isolation.
        # (row-index splitting risks the same content appearing in both splits
        # when the same yt_id is represented by multiple generators.)
        all_yt_ids = sorted(all_by_yt.keys())
        split_yt_ids = set(self._get_split_items(all_yt_ids))

        for yt_id in split_yt_ids:
            gen_caches = all_by_yt[yt_id]
            self._fmc_by_yt[yt_id] = gen_caches
            for gen, cache in gen_caches.items():
                self._fmc_gen_tracks[gen].append((cache, yt_id))
                self._fmc_pool.append(cache)

        logger.info(
            f"  FMC loaded: {len(self._fmc_by_yt)} YouTube IDs, "
            f"{len(self._fmc_pool)} tracks, "
            f"{len(self._fmc_gen_tracks)} generators"
        )

    def _build_fmc_same_prompt_pairs(self):
        """Positive FMC pairs: same YouTube ID (same caption), different generators.

        Analogous to SONICS sibling pairs — same musical content was described,
        different AI systems synthesised it, so they are musically similar but
        neither is a derivative of the other.
        Labels: similarity=0.8, attribution=0.0, both AI.
        """
        count = 0
        for yt_id, gen_caches in self._fmc_by_yt.items():
            gens = sorted(gen_caches.keys())
            if len(gens) < 2:
                continue
            for i in range(len(gens)):
                for j in range(i + 1, len(gens)):
                    cache_a = gen_caches[gens[i]]
                    cache_b = gen_caches[gens[j]]
                    if self._register_pair(cache_a, cache_b):
                        self.pairs.append({
                            "cache_a": cache_a,
                            "cache_b": cache_b,
                            "similarity": 0.8,
                            "is_ai_a": 1.0,
                            "is_ai_b": 1.0,
                            "is_attribution": 0.0,
                            "source": "fmc_same_prompt_pos",
                        })
                        count += 1
        logger.info(f"  FMC same-prompt positive pairs: {count}")

    def _build_fmc_negatives(self):
        """Negative FMC pairs: different generators, different YouTube IDs."""
        gen_tracks = self._fmc_gen_tracks
        gens = [g for g in gen_tracks if len(gen_tracks[g]) >= 2]
        if len(gens) < 2:
            logger.info("  FMC cross-generator negatives: 0")
            return

        count = 0
        n_target = int(sum(len(gen_tracks[g]) for g in gens) * 0.3)
        n_target = max(n_target, 50)

        gen_pairs = [(g1, g2) for g1 in gens for g2 in gens if g1 < g2]
        self.rng.shuffle(gen_pairs)

        for g1, g2 in gen_pairs:
            tracks1 = list(gen_tracks[g1])
            tracks2 = list(gen_tracks[g2])
            self.rng.shuffle(tracks1)
            self.rng.shuffle(tracks2)

            for (cache_a, yt_a), (cache_b, yt_b) in zip(tracks1, tracks2):
                if yt_a == yt_b:
                    continue  # same prompt — this is a positive pair, not a negative
                if self._register_pair(cache_a, cache_b):
                    self.pairs.append({
                        "cache_a": cache_a,
                        "cache_b": cache_b,
                        "similarity": 0.0,
                        "is_ai_a": 1.0,
                        "is_ai_b": 1.0,
                        "is_attribution": 0.0,
                        "source": "fmc_cross_neg",
                    })
                    count += 1
                if count >= n_target:
                    break
            if count >= n_target:
                break

        logger.info(f"  FMC cross-generator negatives: {count}")

    # ── 7. Cross-Dataset Negatives ──

    def _build_cross_dataset_negatives(self):
        """Cross-dataset negatives: SONICS real ↔ FMC fake, SONICS fake ↔ FMC fake."""
        if not self._fmc_pool:
            return

        fmc = list(set(self._fmc_pool))
        reals = list(set(self._real_pool))
        all_fakes = list(set(c for caches in self._fake_by_gen.values() for c in caches))

        self.rng.shuffle(fmc)
        self.rng.shuffle(reals)
        self.rng.shuffle(all_fakes)

        # SONICS real ↔ FMC fake
        count_rf = 0
        n_target = min(len(reals), len(fmc), 200)
        for real, fake in zip(reals, fmc):
            if self._register_pair(real, fake):
                self.pairs.append({
                    "cache_a": real,
                    "cache_b": fake,
                    "similarity": 0.0,
                    "is_ai_a": 0.0,
                    "is_ai_b": 1.0,
                    "is_attribution": 0.0,
                    "source": "cross_real_vs_fmc_neg",
                })
                count_rf += 1
            if count_rf >= n_target:
                break

        # SONICS fake ↔ FMC fake
        count_ff = 0
        n_target = min(len(all_fakes), len(fmc), 200)
        fmc_idx = 0
        for fake in all_fakes:
            if count_ff >= n_target:
                break
            fmc_track = fmc[fmc_idx % len(fmc)]
            fmc_idx += 1
            if self._register_pair(fake, fmc_track):
                self.pairs.append({
                    "cache_a": fake,
                    "cache_b": fmc_track,
                    "similarity": 0.0,
                    "is_ai_a": 1.0,
                    "is_ai_b": 1.0,
                    "is_attribution": 0.0,
                    "source": "cross_sonics_vs_fmc_neg",
                })
                count_ff += 1

        logger.info(
            f"  Cross-dataset negatives: {count_rf} real↔FMC, "
            f"{count_ff} SONICS↔FMC"
        )

    # ── Stats ──

    def _log_pair_stats(self):
        """Log pair distribution by source."""
        sources = Counter(p["source"] for p in self.pairs)
        for src, cnt in sorted(sources.items()):
            logger.info(f"    {src}: {cnt}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        feat_a = _load_features(pair["cache_a"], self.max_chunks)
        feat_b = _load_features(pair["cache_b"], self.max_chunks)

        labels = {
            "similarity": np.float32(pair["similarity"]),
            "is_ai_a": np.float32(pair["is_ai_a"]),
            "is_ai_b": np.float32(pair["is_ai_b"]),
            "is_attribution": np.float32(pair["is_attribution"]),
        }
        return feat_a, feat_b, labels


# ── Validation dataset (Echoes, ATA only) ──────────────────────────────────


class EchoesValDataset(Dataset):
    """Validation dataset from Echoes: audio-to-audio tracks only."""

    def __init__(
        self,
        data_dir: str = "data",
        cache_dir: str = "feature_cache",
        max_chunks: int = 12,
        max_pairs: int = 500,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.max_chunks = max_chunks

        self.path_to_cache = {}
        meta_file = self.cache_dir / "feature_metadata.csv"
        if meta_file.exists():
            df_meta = pd.read_csv(meta_file)
            for _, row in df_meta.iterrows():
                # Use full posix path (same as PairDataset) so suffix-matching works
                # correctly and basename collisions are impossible.
                fname = Path(str(row['audio_path']).replace("\\", "/")).as_posix()
                self.path_to_cache[fname] = row['cache_path']
        else:
            logger.warning(f"No metadata found at {meta_file}. Assuming caches match audio stems.")
            self.path_to_cache = {npz.stem + ".wav": str(npz) for npz in self.cache_dir.glob("*.npz")}
        self.pairs = []

        self._ori_cache = {}
        self._gen_caches = []

        self._build_echoes_pairs()
        self._build_echoes_negatives()

        if len(self.pairs) > max_pairs:
            rng = random.Random(42)
            self.pairs = rng.sample(self.pairs, max_pairs)

    def _resolve_cache(self, audio_path: Path) -> Optional[str]:
        search_path = audio_path.as_posix()
        for k, v in self.path_to_cache.items():
            if k.endswith(search_path) or search_path.endswith(k):
                if Path(v).exists():
                    return v
        return None

    def _build_echoes_pairs(self):
        gen_csv = self.data_dir / "echoes_processed" / "processed_dataset_manifest.csv"
        ori_csv = self.data_dir / "echoes_processed" / "processed_originals_manifest.csv"
        if not gen_csv.exists() or not ori_csv.exists():
            return

        gen_df = pd.read_csv(gen_csv)
        ori_df = pd.read_csv(ori_csv)
        base_dir = self.data_dir / "echoes_processed"

        # Filter to ATA (audio-to-audio) only
        gen_df = gen_df[gen_df["type"] == "ATA"]

        # Build original_audio -> cache_path mapping
        self._ori_cache = {}
        for _, row in ori_df.iterrows():
            pp = row.get("processed_path", "")
            if pd.isna(pp) or not pp:
                continue
            fp = base_dir / pp
            cache = self._resolve_cache(fp)
            if cache:
                ori_name = row.get("original_audio", "")
                if ori_name and not pd.isna(ori_name):
                    self._ori_cache[ori_name] = cache

        # Match generated -> original
        self._gen_caches = []
        for _, row in gen_df.iterrows():
            pp = row.get("processed_path", "")
            ori_name = row.get("original_audio", "")
            if pd.isna(pp) or not pp or pd.isna(ori_name) or not ori_name:
                continue

            fp = base_dir / pp
            cache_b = self._resolve_cache(fp)
            cache_a = self._ori_cache.get(ori_name)

            if cache_a and cache_b:
                self._gen_caches.append(cache_b)
                self.pairs.append({
                    "cache_a": cache_a,
                    "cache_b": cache_b,
                    "similarity": 1.0,
                    "is_ai_a": 0.0,
                    "is_ai_b": 1.0,
                    "is_attribution": 1.0,
                    "source": "echoes_pos",
                })

    def _build_echoes_negatives(self):
        """Random pairings of real originals with unrelated AI fakes."""
        ori_list = list(self._ori_cache.values())
        gen_list = list(set(self._gen_caches))

        if len(ori_list) < 2 or len(gen_list) < 2:
            return

        positive_set = {(p["cache_a"], p["cache_b"]) for p in self.pairs}
        positive_set |= {(p["cache_b"], p["cache_a"]) for p in self.pairs}

        rng = random.Random(42)
        n_pos = len(self.pairs)
        added = 0
        for _ in range(n_pos):
            a = rng.choice(ori_list)
            b = rng.choice(gen_list)
            if (a, b) in positive_set:
                continue
            self.pairs.append({
                "cache_a": a,
                "cache_b": b,
                "similarity": 0.0,
                "is_ai_a": 0.0,
                "is_ai_b": 1.0,
                "is_attribution": 0.0,
                "source": "echoes_neg",
            })
            positive_set.add((a, b))
            added += 1
            if added >= n_pos:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        feat_a = _load_features(pair["cache_a"], self.max_chunks)
        feat_b = _load_features(pair["cache_b"], self.max_chunks)

        labels = {
            "similarity": np.float32(pair["similarity"]),
            "is_ai_a": np.float32(pair["is_ai_a"]),
            "is_ai_b": np.float32(pair["is_ai_b"]),
            "is_attribution": np.float32(pair["is_attribution"]),
        }
        return feat_a, feat_b, labels


# ── Collate function ──────────────────────────────────────────────────


def collate_pairs(batch):
    """Custom collate that pads chunk sequences to the same length within a batch."""
    feat_a_list, feat_b_list, labels_list = zip(*batch)

    def _pad_chunks(feat_list, key="classical_chunks", fallback_dim=432):
        # Safely extract chunks if the key exists, otherwise use empty arrays
        chunks = [
            f[key] if key in f else np.zeros((0, fallback_dim), dtype=np.float32) 
            for f in feat_list
        ]
        
        max_len = max(c.shape[0] for c in chunks)
        
        # If all tracks are missing chunks (e.g., in 'embedding' mode), 
        # return a safe dummy tensor so the forward pass doesn't break.
        if max_len == 0:
            dummy_padded = torch.zeros((len(chunks), 1, fallback_dim), dtype=torch.float32)
            dummy_mask = torch.zeros((len(chunks), 1), dtype=torch.float32)
            return dummy_padded, dummy_mask

        # Find the actual feature dimension from the first non-empty chunk
        dim = next((c.shape[1] for c in chunks if c.shape[0] > 0), fallback_dim)
        
        padded = np.zeros((len(chunks), max_len, dim), dtype=np.float32)
        mask = np.zeros((len(chunks), max_len), dtype=np.float32)
        
        for i, c in enumerate(chunks):
            if c.shape[0] > 0:
                padded[i, : c.shape[0]] = c
                mask[i, : c.shape[0]] = 1.0
                
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _stack_fixed(feat_list, key, fallback_dim):
        arrays = []
        for f in feat_list:
            if key in f:
                arrays.append(f[key])
            else:
                arrays.append(np.zeros(fallback_dim, dtype=np.float32))
        return torch.from_numpy(np.stack(arrays))

    # Apply the safe padding function
    chunks_a, mask_a = _pad_chunks(feat_a_list)
    chunks_b, mask_b = _pad_chunks(feat_b_list)

    batch_dict = {
        "chunks_a": chunks_a,
        "mask_a": mask_a,
        "chunks_b": chunks_b,
        "mask_b": mask_b,
        "classical_a": _stack_fixed(feat_a_list, "classical", 432),
        "classical_b": _stack_fixed(feat_b_list, "classical", 432),
        "ai_det_a": _stack_fixed(feat_a_list, "ai_detection", 22),
        "ai_det_b": _stack_fixed(feat_b_list, "ai_detection", 22),
        "fakeprint_a": _stack_fixed(feat_a_list, "fakeprint", 897),
        "fakeprint_b": _stack_fixed(feat_b_list, "fakeprint", 897),
        "mert_a": _stack_fixed(feat_a_list, "mert", 1024),
        "mert_b": _stack_fixed(feat_b_list, "mert", 1024),
        "clap_a": _stack_fixed(feat_a_list, "clap", 512),
        "clap_b": _stack_fixed(feat_b_list, "clap", 512),
        "lyric_emb_a": _stack_fixed(feat_a_list, "lyric_embedding", 384),
        "lyric_emb_b": _stack_fixed(feat_b_list, "lyric_embedding", 384),
    }

    label_dict = {
        k: torch.tensor([l[k] for l in labels_list], dtype=torch.float32)
        for k in ["similarity", "is_ai_a", "is_ai_b", "is_attribution"]
    }

    return batch_dict, label_dict