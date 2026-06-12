from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ImportError("PocketBindModel requires PyTorch. Install torch before training.") from exc


class PocketBindModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_peptide_len: int = 15,
        max_hla_len: int = 34,
        max_context_len: int = 12,
        num_tasks: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.peptide_pos = nn.Parameter(torch.zeros(1, max_peptide_len, hidden_dim))
        self.hla_pos = nn.Parameter(torch.zeros(1, max_hla_len, hidden_dim))
        self.context_pos = nn.Parameter(torch.zeros(1, max_context_len, hidden_dim))
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.peptide_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hla_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers // 2))

        self.peptide_to_hla = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.hla_to_peptide = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        feature_dim = hidden_dim * 4
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.ba_head = nn.Linear(hidden_dim, 1)
        self.el_head = nn.Linear(hidden_dim, 1)
        self.iedb_head = nn.Linear(hidden_dim, 1)
        self.cedar_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        return mask == 0

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    def forward(
        self,
        *,
        peptide_ids: torch.Tensor,
        peptide_mask: torch.Tensor,
        hla_ids: torch.Tensor,
        hla_mask: torch.Tensor,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        task_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        peptide = self.embedding(peptide_ids) + self.peptide_pos[:, : peptide_ids.shape[1]]
        hla = self.embedding(hla_ids) + self.hla_pos[:, : hla_ids.shape[1]]
        context = self.embedding(context_ids) + self.context_pos[:, : context_ids.shape[1]]

        peptide = self.peptide_encoder(peptide, src_key_padding_mask=self._key_padding_mask(peptide_mask))
        hla = self.hla_encoder(hla, src_key_padding_mask=self._key_padding_mask(hla_mask))
        context = self.context_encoder(context, src_key_padding_mask=self._key_padding_mask(context_mask))

        pep_cross, _ = self.peptide_to_hla(
            peptide,
            hla,
            hla,
            key_padding_mask=self._key_padding_mask(hla_mask),
            need_weights=False,
        )
        hla_cross, _ = self.hla_to_peptide(
            hla,
            peptide,
            peptide,
            key_padding_mask=self._key_padding_mask(peptide_mask),
            need_weights=False,
        )
        peptide = self.norm(peptide + pep_cross)
        hla = self.norm(hla + hla_cross)

        pooled = torch.cat(
            [
                self._masked_mean(peptide, peptide_mask),
                self._masked_mean(hla, hla_mask),
                self._masked_mean(context, context_mask),
                self.task_embedding(task_id),
            ],
            dim=-1,
        )
        hidden = self.shared(pooled)
        return {
            "ba": self.ba_head(hidden).squeeze(-1),
            "el": self.el_head(hidden).squeeze(-1),
            "iedb": self.iedb_head(hidden).squeeze(-1),
            "cedar": self.cedar_head(hidden).squeeze(-1),
        }

