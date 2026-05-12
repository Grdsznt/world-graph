"""Transition Model: the core world model T(s_t, a_t) → s_{t+1}.

Given the current scene graph state and a high-level action, predicts
the next scene graph state including:
- Node feature changes (positions, states, properties)
- Edge topology changes (add/remove edges)
- Node existence changes (objects appearing/disappearing)
"""

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj

from worldgraph.config import TransitionModelConfig
from worldgraph.world_model.encoder import SceneGraphEncoder
from worldgraph.world_model.action_encoder import ActionEncoder


class TransitionOutput(NamedTuple):
    """Output of the transition model."""
    predicted_node_features: torch.Tensor  # [N, node_feat_dim] — predicted next-state features
    edge_logits: torch.Tensor  # [N, N] — logits for edge existence (per graph)
    node_existence_logits: torch.Tensor  # [N, 1] — logits for node still existing
    node_embeddings: torch.Tensor  # [N, hidden_dim] — intermediate node representations


class ActionConditioner(nn.Module):
    """Conditions node representations on the action via cross-attention.

    Each node attends to the action embedding, allowing the model to learn
    which nodes are affected by the action (e.g., "pick up mug" should
    primarily modify the mug node and its connected surfaces).
    """

    def __init__(self, hidden_dim: int, action_dim: int, num_heads: int = 8):
        super().__init__()
        # Project action to hidden dim
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention: nodes (Q) attend to action (KV)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Post-attention FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_emb: torch.Tensor,
        action_emb: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Condition node embeddings on the action.

        Args:
            node_emb: [N_total, hidden_dim] node embeddings from graph encoder.
            action_emb: [B, action_dim] action embeddings from CLIP.
            batch: [N_total] batch assignment vector.

        Returns:
            [N_total, hidden_dim] action-conditioned node embeddings.
        """
        # Project action
        action_proj = self.action_proj(action_emb)  # [B, hidden_dim]

        # Convert to dense batch for cross-attention
        node_dense, mask = to_dense_batch(node_emb, batch)  # [B, max_N, D]
        key_padding_mask = ~mask

        # Action as KV: expand to [B, 1, hidden_dim]
        action_kv = action_proj.unsqueeze(1)  # [B, 1, D]

        # Cross-attention: each node queries the action
        attn_out, _ = self.cross_attn(
            query=node_dense,
            key=action_kv,
            value=action_kv,
        )  # [B, max_N, D]

        # Back to sparse
        attn_sparse = attn_out[mask]
        node_emb = self.norm(node_emb + attn_sparse)

        # FFN
        node_emb = self.ffn_norm(node_emb + self.ffn(node_emb))

        return node_emb


class NodeFeaturePredictionHead(nn.Module):
    """Predicts the change in node features (residual prediction).

    Outputs a delta that is added to the original features to produce
    the predicted next-state features: x_{t+1} = x_t + Δx.
    """

    def __init__(self, hidden_dim: int, node_feat_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, node_feat_dim),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """Predict node feature deltas.

        Args:
            node_emb: [N, hidden_dim] action-conditioned node embeddings.

        Returns:
            [N, node_feat_dim] predicted feature deltas.
        """
        return self.head(node_emb)


class EdgePredictionHead(nn.Module):
    """Predicts edge existence between all pairs of nodes.

    Uses bilinear-style scoring: score(i, j) = MLP(concat(h_i, h_j)).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Predict pairwise edge existence logits.

        Args:
            node_emb: [N_total, hidden_dim] node embeddings.
            batch: [N_total] batch assignment.

        Returns:
            List of [n_i, n_i] edge logit matrices, one per graph in batch.
        """
        # Convert to dense for pairwise computation
        node_dense, mask = to_dense_batch(node_emb, batch)  # [B, max_N, D]
        B, max_N, D = node_dense.shape

        # Pairwise features: concat(h_i, h_j) for all pairs
        h_i = node_dense.unsqueeze(2).expand(-1, -1, max_N, -1)  # [B, N, N, D]
        h_j = node_dense.unsqueeze(1).expand(-1, max_N, -1, -1)  # [B, N, N, D]
        pair_feats = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2D]

        # Score each pair
        edge_logits = self.edge_mlp(pair_feats).squeeze(-1)  # [B, N, N]

        # Mask out invalid positions (padding nodes)
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # [B, N, N]
        edge_logits = edge_logits.masked_fill(~pair_mask, float("-inf"))

        return edge_logits, mask


class NodeExistenceHead(nn.Module):
    """Predicts whether each node still exists in the next state.

    Handles objects being removed from the scene (e.g., picked up and
    carried away, consumed in cooking, etc.).
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        """Predict node existence logits.

        Args:
            node_emb: [N, hidden_dim] node embeddings.

        Returns:
            [N, 1] existence logits (positive = exists).
        """
        return self.head(node_emb)


class TransitionModel(nn.Module):
    """World Model Transition Function: T(s_t, a_t) → s_{t+1}.

    The core component that predicts how the scene graph changes
    when a high-level action is executed.

    Architecture:
    1. Encode current scene graph with GPS (GATv2 + Transformer)
    2. Condition node representations on the action via cross-attention
    3. Predict node feature deltas (residual)
    4. Predict edge existence (pairwise scoring)
    5. Predict node existence (per-node classification)
    """

    def __init__(self, config: TransitionModelConfig):
        super().__init__()
        self.config = config
        node_feat_dim = config.node_encoder.total_feat_dim
        hidden_dim = config.graph_encoder.hidden_dim
        action_dim = config.action_feat_dim

        # Scene graph encoder
        self.graph_encoder = SceneGraphEncoder(
            config=config.graph_encoder,
            input_dim=node_feat_dim,
        )

        # Action conditioner (cross-attention)
        self.action_conditioner = ActionConditioner(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            num_heads=config.cross_attn_heads,
        )

        # Prediction heads
        self.node_delta_head = NodeFeaturePredictionHead(
            hidden_dim=hidden_dim,
            node_feat_dim=node_feat_dim,
        )

        if config.predict_edge_changes:
            self.edge_head = EdgePredictionHead(hidden_dim=hidden_dim)
        else:
            self.edge_head = None

        if config.predict_node_existence:
            self.existence_head = NodeExistenceHead(hidden_dim=hidden_dim)
        else:
            self.existence_head = None

    def forward(
        self,
        graph_data: Data,
        action_emb: torch.Tensor,
    ) -> TransitionOutput:
        """Predict the next scene graph state.

        Args:
            graph_data: PyG Data or Batch with:
                - x: [N, node_feat_dim] node features
                - edge_index: [2, E] edge indices
                - edge_attr: [E] edge type indices
                - batch: [N] batch vector (auto-created by Batch)
            action_emb: [B, action_dim] action embeddings.

        Returns:
            TransitionOutput with predicted next-state features, edges, existence.
        """
        # Get batch vector
        if hasattr(graph_data, "batch") and graph_data.batch is not None:
            batch = graph_data.batch
        else:
            batch = torch.zeros(graph_data.x.size(0), dtype=torch.long,
                                device=graph_data.x.device)

        # 1. Encode current scene graph
        node_emb = self.graph_encoder(
            x=graph_data.x,
            edge_index=graph_data.edge_index,
            edge_attr=graph_data.edge_attr,
            batch=batch,
        )  # [N, hidden_dim]

        # 2. Condition on action
        node_emb = self.action_conditioner(node_emb, action_emb, batch)

        # 3. Predict node feature changes (residual)
        delta = self.node_delta_head(node_emb)  # [N, node_feat_dim]
        predicted_features = graph_data.x + delta

        # 4. Predict edge changes
        if self.edge_head is not None:
            edge_logits, edge_mask = self.edge_head(node_emb, batch)
        else:
            edge_logits = None

        # 5. Predict node existence
        if self.existence_head is not None:
            existence_logits = self.existence_head(node_emb)
        else:
            existence_logits = None

        return TransitionOutput(
            predicted_node_features=predicted_features,
            edge_logits=edge_logits,
            node_embeddings=node_emb,
            node_existence_logits=existence_logits,
        )

    def predict_next_graph(
        self,
        graph_data: Data,
        action_emb: torch.Tensor,
        edge_threshold: float = 0.5,
    ) -> Data:
        """Predict and materialize the next scene graph as a new PyG Data.

        This is the user-facing method for imagination: given current state
        and action, produce a concrete predicted next-state graph.

        Args:
            graph_data: Current scene graph (single graph, not batched).
            action_emb: [1, action_dim] action embedding.
            edge_threshold: Probability threshold for edge existence.

        Returns:
            New PyG Data representing predicted s_{t+1}.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(graph_data, action_emb)

            # Apply node existence filter
            if output.node_existence_logits is not None:
                exist_probs = torch.sigmoid(output.node_existence_logits.squeeze(-1))
                alive_mask = exist_probs > 0.5
            else:
                alive_mask = torch.ones(
                    graph_data.x.size(0), dtype=torch.bool,
                    device=graph_data.x.device,
                )

            # Filter to surviving nodes
            new_x = output.predicted_node_features[alive_mask]

            # Build new edge index from edge predictions
            if output.edge_logits is not None:
                edge_probs = torch.sigmoid(output.edge_logits[0])  # [N, N]
                # Filter to alive nodes
                alive_indices = alive_mask.nonzero(as_tuple=True)[0]
                n_alive = alive_indices.size(0)

                # Sub-select edge probs for alive nodes
                sub_probs = edge_probs[alive_indices][:, alive_indices]

                # Threshold to get edges
                edge_mask = sub_probs > edge_threshold
                # Remove self-loops
                edge_mask.fill_diagonal_(False)
                src, tgt = edge_mask.nonzero(as_tuple=True)
                new_edge_index = torch.stack([src, tgt], dim=0)

                # Edge attributes (use UNKNOWN type for predicted edges)
                new_edge_attr = torch.full(
                    (new_edge_index.size(1),), 11,  # EdgeType.UNKNOWN
                    dtype=torch.long, device=graph_data.x.device,
                )
            else:
                # Keep original edges between surviving nodes
                new_edge_index = graph_data.edge_index
                new_edge_attr = graph_data.edge_attr

            # Construct new graph
            new_data = Data(
                x=new_x,
                edge_index=new_edge_index,
                edge_attr=new_edge_attr,
                num_nodes=new_x.size(0),
            )

            # Carry over positions if available
            if hasattr(graph_data, "pos") and graph_data.pos is not None:
                new_data.pos = graph_data.pos[alive_mask]

            if hasattr(graph_data, "semantic_labels"):
                new_data.semantic_labels = graph_data.semantic_labels[alive_mask]

            if hasattr(graph_data, "label_names"):
                names = graph_data.label_names
                new_data.label_names = [
                    names[i] for i in alive_indices.cpu().tolist()
                ]

            return new_data
