"""
Attribution model for AI-generated music detection and pairwise similarity.

Architecture:
    Track Encoder (Siamese, shared weights):
        Per-chunk classical features [n_chunks, 432]
        → Linear projection → Transformer encoder → Attention pooling
        → track_emb [hidden_dim]

    Optionally concatenates mean-pooled MERT (1024) and CLAP (512) embeddings.

    AI Detection Head (per-track):
        ai_detection [22] + fakeprint [897] → MLP → ai_index [1]

    Similarity Head (pairwise):
        [emb_a, emb_b, |a-b|, a*b] → MLP → similarity_score [1]

    Attribution Head:
        [similarity, ai_a, ai_b, emb_a, emb_b] → MLP → attribution_score [1]

Outputs:
    {
        "similarity_score": float,   # Musical similarity (0-1)
        "ai_index_a": float,         # P(track A is AI-generated)
        "ai_index_b": float,         # P(track B is AI-generated)
        "attribution_score": float,  # P(B is derived from A)
    }
"""

from .network import AttributionModel
from .dataset import PairDataset, EchoesValDataset

__all__ = ["AttributionModel", "PairDataset", "EchoesValDataset"]
