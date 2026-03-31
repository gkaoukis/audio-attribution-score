"""
Multi-head attribution model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LYRIC_FEAT_DIM = 64   # output dimension of LyricEncoder


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over a sequence."""
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.query(x).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)  # [B, T]
        return (weights.unsqueeze(-1) * x).sum(dim=1)  # [B, D]


class ChunkEncoder(nn.Module):
    """Encodes variable-length chunk sequences into fixed-size track embeddings."""
    def __init__(
        self,
        input_dim: int = 432,
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_chunks: int = 16,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Embedding(max_chunks, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = AttentionPooling(hidden_dim)

    def forward(self, chunks: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = chunks.shape
        x = self.proj(chunks)

        positions = torch.arange(T, device=chunks.device).unsqueeze(0).expand(B, T)
        positions = positions.clamp(max=self.pos_embed.num_embeddings - 1)
        x = x + self.pos_embed(positions)

        src_key_padding_mask = mask == 0
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        return self.pool(x, mask)


class TrackEncoder(nn.Module):
    """Full track encoder: chunk encoder + optional embedding fusion."""
    def __init__(
        self,
        hidden_dim: int = 256,
        use_classical: bool = True,
        use_embeddings: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_classical = use_classical
        self.use_embeddings = use_embeddings

        if self.use_classical:
            self.chunk_encoder = ChunkEncoder(
                input_dim=432, hidden_dim=hidden_dim, dropout=dropout
            )

        if self.use_embeddings:
            self.emb_proj = nn.Sequential(
                nn.LayerNorm(1024 + 512),
                nn.Linear(1024 + 512, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        if self.use_classical and self.use_embeddings:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, 1),
                nn.Sigmoid(),
            )
            self.fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

    def forward(
        self,
        chunks: torch.Tensor,
        mask: torch.Tensor,
        mert: torch.Tensor = None,
        clap: torch.Tensor = None,
    ) -> torch.Tensor:

        chunk_emb = None
        if self.use_classical:
            chunk_emb = self.chunk_encoder(chunks, mask)

        emb_proj = None
        if self.use_embeddings and mert is not None and clap is not None:
            emb_cat = torch.cat([mert, clap], dim=-1)
            emb_proj = self.emb_proj(emb_cat)

        if self.use_classical and self.use_embeddings:
            combined = torch.cat([chunk_emb, emb_proj], dim=-1)
            gate_val = self.gate(combined)
            fused = self.fuse(combined)
            return gate_val * chunk_emb + (1 - gate_val) * fused
        elif self.use_classical:
            return chunk_emb
        elif self.use_embeddings:
            return emb_proj
        else:
            raise ValueError("TrackEncoder must use classical features, embeddings, or both.")


class AIDetectionHead(nn.Module):
    """Per-track AI detection from track embeddings and optional artifact features."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, track_emb: torch.Tensor, ai_feats: torch.Tensor = None) -> torch.Tensor:
        if ai_feats is not None:
            x = torch.cat([track_emb, ai_feats], dim=-1)
        else:
            x = track_emb
        return self.net(x)


class SimilarityHead(nn.Module):
    """Pairwise similarity from track embeddings."""
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        interaction = torch.cat([
            emb_a,
            emb_b,
            torch.abs(emb_a - emb_b),
            emb_a * emb_b,
        ], dim=-1)
        return self.net(interaction)


class LyricEncoder(nn.Module):
    """Projects a pair of SBERT lyric embeddings into a compact interaction vector.

    Concatenates [emb_a, emb_b, |emb_a - emb_b|, emb_a * emb_b] and projects
    to LYRIC_FEAT_DIM, giving the attribution head richer lyric signal than a
    single cosine similarity scalar.
    """
    def __init__(
        self,
        emb_dim: int = 384,
        out_dim: int = LYRIC_FEAT_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim * 4),
            nn.Linear(emb_dim * 4, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
            nn.GELU(),
        )

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        interaction = torch.cat([
            emb_a,
            emb_b,
            torch.abs(emb_a - emb_b),
            emb_a * emb_b,
        ], dim=-1)
        return self.net(interaction)


class AttributionHead(nn.Module):
    """Attribution score combining similarity, AI detection, track embeddings, and lyrics."""
    def __init__(self, hidden_dim: int = 256, use_ai: bool = True,
                 use_lyrics: bool = False, dropout: float = 0.1):
        super().__init__()
        self.use_ai = use_ai
        self.use_lyrics = use_lyrics

        # sim_logit (1) + emb_a + emb_b (hidden_dim * 2)
        input_dim = 1 + hidden_dim * 2
        if use_ai:
            input_dim += 2  # ai_logit_a, ai_logit_b
        if use_lyrics:
            input_dim += LYRIC_FEAT_DIM  # lyric interaction vector

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        sim_logit: torch.Tensor,
        ai_logit_a: torch.Tensor,
        ai_logit_b: torch.Tensor,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        lyric_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        parts = [sim_logit, emb_a, emb_b]
        if self.use_ai:
            parts.insert(1, ai_logit_a)
            parts.insert(2, ai_logit_b)
        if self.use_lyrics and lyric_feat is not None:
            parts.append(lyric_feat)
        x = torch.cat(parts, dim=-1)
        return self.net(x)


class AttributionModel(nn.Module):
    """Full multi-task model for attribution, similarity, and AI detection."""
    def __init__(
        self,
        hidden_dim: int = 256,
        feature_set: str = "advanced",
        use_lyrics: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_set = feature_set

        if feature_set == "mix":
            self.use_lyrics = True
        else:
            self.use_lyrics = use_lyrics

        self.use_classical = feature_set in ("basic", "advanced", "mix")
        self.use_embeddings = feature_set in ("embedding", "advanced", "mix")
        self.use_artifacts = feature_set in ("advanced", "mix")

        self.track_encoder = TrackEncoder(
            hidden_dim=hidden_dim,
            use_classical=self.use_classical,
            use_embeddings=self.use_embeddings,
            dropout=dropout,
        )
        self.sim_head = SimilarityHead(hidden_dim=hidden_dim, dropout=dropout)

        ai_input_dim = hidden_dim
        if self.use_artifacts:
            ai_input_dim += 919  # 22 (ai_detection) + 897 (fakeprint)

        self.ai_head = AIDetectionHead(input_dim=ai_input_dim, hidden_dim=128, dropout=dropout)

        if self.use_lyrics:
            self.lyric_encoder = LyricEncoder(emb_dim=384, out_dim=LYRIC_FEAT_DIM, dropout=dropout)
        else:
            self.lyric_encoder = None

        self.attr_head = AttributionHead(
            hidden_dim=hidden_dim,
            use_ai=True,
            use_lyrics=self.use_lyrics,
            dropout=dropout,
        )

    def forward(self, batch: dict) -> dict:
        mert_a = batch.get("mert_a") if self.use_embeddings else None
        clap_a = batch.get("clap_a") if self.use_embeddings else None
        mert_b = batch.get("mert_b") if self.use_embeddings else None
        clap_b = batch.get("clap_b") if self.use_embeddings else None

        emb_a = self.track_encoder(batch["chunks_a"], batch["mask_a"], mert_a, clap_a)
        emb_b = self.track_encoder(batch["chunks_b"], batch["mask_b"], mert_b, clap_b)

        sim_logit = self.sim_head(emb_a, emb_b)

        if self.use_artifacts:
            ai_feats_a = torch.cat([batch["ai_det_a"], batch["fakeprint_a"]], dim=-1)
            ai_feats_b = torch.cat([batch["ai_det_b"], batch["fakeprint_b"]], dim=-1)
        else:
            ai_feats_a = None
            ai_feats_b = None

        ai_logit_a = self.ai_head(emb_a, ai_feats_a)
        ai_logit_b = self.ai_head(emb_b, ai_feats_b)

        lyric_feat = None
        if self.use_lyrics and self.lyric_encoder is not None:
            lyric_emb_a = batch.get("lyric_emb_a")
            lyric_emb_b = batch.get("lyric_emb_b")
            if lyric_emb_a is not None and lyric_emb_b is not None:
                lyric_feat = self.lyric_encoder(lyric_emb_a, lyric_emb_b)

        attr_logit = self.attr_head(
            sim_logit, ai_logit_a, ai_logit_b, emb_a, emb_b, lyric_feat
        )

        return {
            "sim_logit": sim_logit,
            "ai_logit_a": ai_logit_a,
            "ai_logit_b": ai_logit_b,
            "attr_logit": attr_logit,
            "emb_a": emb_a,
            "emb_b": emb_b,
        }

    def predict(self, batch: dict) -> dict:
        with torch.no_grad():
            out = self.forward(batch)
        return {
            "similarity_score": torch.sigmoid(out["sim_logit"]).squeeze(-1),
            "ai_index_a": torch.sigmoid(out["ai_logit_a"]).squeeze(-1),
            "ai_index_b": torch.sigmoid(out["ai_logit_b"]).squeeze(-1),
            "attribution_score": torch.sigmoid(out["attr_logit"]).squeeze(-1),
        }
