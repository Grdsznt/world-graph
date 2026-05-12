"""GPS (General, Powerful, Scalable) Graph Transformer encoder.

Combines GATv2 local message passing with global Transformer self-attention
at each layer, with edge-type-aware attention.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from worldgraph.config import GraphEncoderConfig


class EdgeTypeEmbedding(nn.Module):
    """Learnable embeddings for edge types (isOn, isIn, nearBy, etc.)."""

    def __init__(self, num_types: int = 12, emb_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_types + 1, emb_dim, padding_idx=0)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Encode edge types.

        Args:
            edge_attr: [E] tensor of edge type indices.

        Returns:
            [E, emb_dim] edge embeddings.
        """
        return self.embedding(edge_attr)


class GPSBlock(nn.Module):
    """A single GPS block: local GATv2 + global Transformer + FFN.

    The hybrid approach captures both local graph structure (via GATv2)
    and global context (via multi-head self-attention).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        gat_heads: int = 8,
        edge_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Local: GATv2 with edge features
        self.local_conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // gat_heads,
            heads=gat_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=True,
        )
        self.local_norm = nn.LayerNorm(hidden_dim)

        # Global: Multi-head self-attention (over all nodes)
        self.global_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through one GPS block.

        Args:
            x: [N_total, hidden_dim] node features (all graphs in batch).
            edge_index: [2, E_total] edge indices.
            edge_attr: [E_total, edge_dim] edge features.
            batch: [N_total] batch assignment vector.

        Returns:
            [N_total, hidden_dim] updated node features.
        """
        # 1. Local message passing (GATv2)
        local_out = self.local_conv(x, edge_index, edge_attr=edge_attr)
        x = self.local_norm(x + local_out)

        # 2. Global self-attention
        # Convert sparse batch to dense for attention: [B, max_N, D]
        x_dense, mask = to_dense_batch(x, batch)  # [B, max_N, D], [B, max_N]
        # Invert mask for attention (True = ignored positions)
        key_padding_mask = ~mask

        global_out, _ = self.global_attn(
            x_dense, x_dense, x_dense,
            key_padding_mask=key_padding_mask,
        )
        # Convert back to sparse: [N_total, D]
        global_out_sparse = global_out[mask]
        x = self.global_norm(x + global_out_sparse)

        # 3. FFN
        x = self.ffn_norm(x + self.ffn(x))

        return x


class SceneGraphEncoder(nn.Module):
    """GPS-based encoder for 3D scene graphs.

    Stacks multiple GPS blocks to produce rich per-node representations
    that capture both local spatial structure and global scene context.
    """

    def __init__(self, config: GraphEncoderConfig, input_dim: int = 1216):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        # Edge type embedding
        self.edge_encoder = EdgeTypeEmbedding(
            num_types=config.num_edge_types,
            emb_dim=config.edge_emb_dim,
        )

        # GPS blocks
        self.blocks = nn.ModuleList([
            GPSBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                gat_heads=config.gat_heads,
                edge_dim=config.edge_emb_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of scene graphs.

        Args:
            x: [N_total, input_dim] raw node features.
            edge_index: [2, E_total] edge indices.
            edge_attr: [E_total] edge type indices.
            batch: [N_total] batch assignment vector.

        Returns:
            [N_total, hidden_dim] encoded node representations.
        """
        # Project input features
        h = self.input_proj(x)

        # Encode edge types
        edge_emb = self.edge_encoder(edge_attr)

        # Pass through GPS blocks
        for block in self.blocks:
            h = block(h, edge_index, edge_emb, batch)

        return h

    def forward_graph_level(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode and pool to graph-level representation.

        Returns:
            [B, hidden_dim] graph-level representations.
        """
        node_emb = self.forward(x, edge_index, edge_attr, batch)
        return global_mean_pool(node_emb, batch)
