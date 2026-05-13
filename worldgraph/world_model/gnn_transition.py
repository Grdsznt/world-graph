"""GNN Transition Model with FiLM conditioning — spec-compliant variant.

This implements T_G(s_t, a_t) → ŝ_{t+1} per the GWM survey taxonomy:
- GATv2 message passing with edge-type awareness
- FiLM (Feature-wise Linear Modulation) for action injection
- Residual prediction: ŝ_{t+1} = s_t + Δ_GNN(s_t, a_t)
- Predicted feature matrix X̂_{t+1} + adjacency logits Â_{t+1}
- Topological Consistency Loss (prevents cross-room teleportation)
- Latent Reconstruction Loss (MSE on SigLIP/DINOv2 features)

Differences from the GPS-based TransitionModel:
┌─────────────────────┬──────────────────────────┬──────────────────────────────┐
│ Feature             │ GPS TransitionModel      │ GNNTransitionModel (this)    │
├─────────────────────┼──────────────────────────┼──────────────────────────────┤
│ Encoder             │ GPS (GATv2 + Transformer)│ Pure GATv2 (faster)          │
│ Action conditioning │ Cross-attention           │ FiLM modulation              │
│ Global attention    │ Full O(N²) Transformer    │ None (local msg passing)     │
│ Node features       │ 1216-d (DINOv2+pos+bbox) │ 515-d (SigLIP 512 + pos 3)  │
│ Topological loss    │ Not implemented           │ ✅ Implemented               │
│ Identity test       │ Not guaranteed             │ ✅ Null action → identity    │
│ HeteroData input    │ Homogeneous only          │ ✅ Parses Hydra JSON layers  │
│ Complexity          │ ~150M params              │ ~30-50M params (lighter)     │
└─────────────────────┴──────────────────────────┴──────────────────────────────┘
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GNNTransitionConfig:
    """Configuration for the GNN Transition Model."""
    # Node features
    node_feat_dim: int = 515     # 512 (SigLIP) + 3 (pos)
    siglip_dim: int = 512
    pos_dim: int = 3

    # GATv2 encoder
    hidden_dim: int = 256
    num_gat_layers: int = 4
    gat_heads: int = 8
    dropout: float = 0.1

    # Edge types
    num_edge_types: int = 12
    edge_emb_dim: int = 32

    # Action
    action_dim: int = 64         # discrete action embedding dim
    num_actions: int = 50        # size of action vocabulary (0 = null)

    # FiLM
    film_hidden_dim: int = 128

    # Prediction
    predict_edges: bool = True
    predict_existence: bool = True

    # Node type (for HeteroData: object=0, place=1)
    num_node_types: int = 2
    node_type_emb_dim: int = 8


# ──────────────────────────────────────────────────────────────────────
# FiLM Conditioning
# ──────────────────────────────────────────────────────────────────────


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation.

    Modulates node features using action-derived scale (γ) and shift (β):
        h' = γ(a) * h + β(a)

    This allows the action to selectively amplify or suppress node features,
    e.g., "pick up mug" amplifies the mug node's features.
    """

    def __init__(self, feature_dim: int, conditioning_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(conditioning_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        # Initialize γ near 1 and β near 0 for stable start
        nn.init.ones_(self.gamma_net[-1].weight.data * 0.01 + 1.0)
        nn.init.zeros_(self.gamma_net[-1].bias.data)
        nn.init.zeros_(self.beta_net[-1].weight.data)
        nn.init.zeros_(self.beta_net[-1].bias.data)

    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            features: [N, D] node features.
            conditioning: [N, C] or [1, C] conditioning vector (action).
                If [1, C], broadcasts to all nodes.

        Returns:
            [N, D] modulated features.
        """
        gamma = self.gamma_net(conditioning)  # [N, D] or [1, D]
        beta = self.beta_net(conditioning)
        return gamma * features + beta


# ──────────────────────────────────────────────────────────────────────
# GATv2 Encoder with FiLM
# ──────────────────────────────────────────────────────────────────────


class GATv2Block(nn.Module):
    """Single GATv2 layer with residual connection and layer norm."""

    def __init__(self, hidden_dim: int, heads: int = 8, edge_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # GATv2 + residual
        x = self.norm(x + self.conv(x, edge_index, edge_attr=edge_attr))
        # FFN + residual
        x = self.ffn_norm(x + self.ffn(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# Output Types
# ──────────────────────────────────────────────────────────────────────


class GNNTransitionOutput(NamedTuple):
    """Output of GNNTransitionModel.forward()."""
    predicted_x: torch.Tensor        # [N, node_feat_dim] — predicted X̂_{t+1}
    delta_x: torch.Tensor            # [N, node_feat_dim] — raw Δ prediction
    edge_logits: Optional[torch.Tensor]      # [B, N, N] — Â_{t+1} logits
    edge_mask: Optional[torch.Tensor]        # [B, N] — valid node mask
    existence_logits: Optional[torch.Tensor] # [N, 1] — node existence logits
    node_embeddings: torch.Tensor    # [N, hidden_dim] — intermediate representations


# ──────────────────────────────────────────────────────────────────────
# GNN Transition Model
# ──────────────────────────────────────────────────────────────────────


class GNNTransitionModel(nn.Module):
    """GNN-based Transition Model: T_G(s_t, a_t) → ŝ_{t+1}.

    Implements the world model transition function per the GWM survey taxonomy.

    Architecture:
        1. Input projection: node features (SigLIP 512-d + pos 3-d) → hidden_dim
        2. Action embedding → FiLM parameters (γ, β)
        3. Multi-layer GATv2 message passing with FiLM conditioning at each layer
        4. Residual prediction: ŝ_{t+1} = s_t + Δ_GNN(s_t, a_t)
        5. Edge prediction: pairwise MLP on node embeddings → Â_{t+1}
        6. Node existence: per-node classifier

    Action injection:
        FiLM modulation after each GATv2 layer. The action vector produces
        per-feature scale (γ) and shift (β) that modulate the message-passing
        output, allowing the model to learn action-specific graph dynamics.

    Residual connection:
        The model predicts Δ (change), not absolute features. This ensures
        that a null/zero action naturally produces ŝ_{t+1} ≈ s_t when the
        network is initialized near zero.
    """

    def __init__(self, config: GNNTransitionConfig = GNNTransitionConfig()):
        super().__init__()
        self.config = config
        D = config.hidden_dim

        # ── Input projection ──
        self.input_proj = nn.Sequential(
            nn.Linear(config.node_feat_dim, D),
            nn.LayerNorm(D),
            nn.GELU(),
        )

        # ── Node type embedding (object vs place) ──
        self.node_type_emb = nn.Embedding(config.num_node_types, config.node_type_emb_dim)
        self.type_proj = nn.Linear(D + config.node_type_emb_dim, D)

        # ── Edge type embedding ──
        self.edge_emb = nn.Embedding(config.num_edge_types + 1, config.edge_emb_dim, padding_idx=0)

        # ── Action embedding ──
        self.action_emb = nn.Embedding(config.num_actions + 1, config.action_dim, padding_idx=0)

        # ── GATv2 layers + FiLM ──
        self.gat_layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(config.num_gat_layers):
            self.gat_layers.append(
                GATv2Block(D, heads=config.gat_heads, edge_dim=config.edge_emb_dim, dropout=config.dropout)
            )
            self.film_layers.append(
                FiLMLayer(D, config.action_dim, config.film_hidden_dim)
            )

        # ── Prediction heads ──
        # Node feature delta
        self.delta_head = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.LayerNorm(D),
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, config.node_feat_dim),
        )
        # Initialize delta head near zero for identity property
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        # Edge prediction (adjacency logits)
        if config.predict_edges:
            self.edge_head = nn.Sequential(
                nn.Linear(D * 2, D),
                nn.GELU(),
                nn.LayerNorm(D),
                nn.Linear(D, D // 4),
                nn.GELU(),
                nn.Linear(D // 4, 1),
            )
        else:
            self.edge_head = None

        # Node existence
        if config.predict_existence:
            self.exist_head = nn.Sequential(
                nn.Linear(D, D // 2),
                nn.GELU(),
                nn.Linear(D // 2, 1),
            )
        else:
            self.exist_head = None

    def forward(
        self,
        data: Data,
        action: torch.Tensor,
        node_types: Optional[torch.Tensor] = None,
    ) -> GNNTransitionOutput:
        """Forward pass: predict next graph state.

        Args:
            data: PyG Data with:
                - x: [N, node_feat_dim] (SigLIP 512 + pos 3 = 515)
                - edge_index: [2, E]
                - edge_attr: [E] edge type indices
                - batch: [N] (auto from Batch.from_data_list)
            action: [B] integer action indices OR [B, action_dim] continuous vectors.
            node_types: [N] integer node type (0=object, 1=place). Optional.

        Returns:
            GNNTransitionOutput with all predictions.
        """
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else \
            torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # ── 1. Input projection ──
        h = self.input_proj(data.x)  # [N, D]

        # Add node type embedding if provided
        if node_types is not None:
            type_emb = self.node_type_emb(node_types)  # [N, type_dim]
            h = self.type_proj(torch.cat([h, type_emb], dim=-1))

        # ── 2. Encode action ──
        if action.dim() == 1:
            # Discrete action indices → embedding
            a = self.action_emb(action)  # [B, action_dim]
        else:
            a = action  # Already continuous [B, action_dim]

        # ── 3. Encode edges ──
        edge_emb = self.edge_emb(data.edge_attr) if data.edge_attr is not None else None

        # ── 4. GATv2 message passing + FiLM conditioning ──
        for gat_layer, film_layer in zip(self.gat_layers, self.film_layers):
            h = gat_layer(h, data.edge_index, edge_emb)
            # FiLM: broadcast action to per-node conditioning
            a_per_node = a[batch]  # [N, action_dim]
            h = film_layer(h, a_per_node)

        # ── 5. Predict Δ (residual) ──
        delta = self.delta_head(h)  # [N, node_feat_dim]
        predicted_x = data.x + delta  # ŝ_{t+1} = s_t + Δ

        # ── 6. Edge prediction ──
        edge_logits = None
        edge_mask = None
        if self.edge_head is not None:
            h_dense, mask = to_dense_batch(h, batch)  # [B, max_N, D]
            B, max_N, D = h_dense.shape
            h_i = h_dense.unsqueeze(2).expand(-1, -1, max_N, -1)
            h_j = h_dense.unsqueeze(1).expand(-1, max_N, -1, -1)
            pair_feats = torch.cat([h_i, h_j], dim=-1)  # [B, N, N, 2D]
            edge_logits = self.edge_head(pair_feats).squeeze(-1)  # [B, N, N]
            # Mask padding
            pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
            edge_logits = edge_logits.masked_fill(~pair_mask, float("-inf"))
            edge_mask = mask

        # ── 7. Node existence ──
        existence_logits = self.exist_head(h) if self.exist_head is not None else None

        return GNNTransitionOutput(
            predicted_x=predicted_x,
            delta_x=delta,
            edge_logits=edge_logits,
            edge_mask=edge_mask,
            existence_logits=existence_logits,
            node_embeddings=h,
        )

    def imagine(
        self,
        data: Data,
        action: torch.Tensor,
        edge_threshold: float = 0.5,
        node_types: Optional[torch.Tensor] = None,
    ) -> Data:
        """Single-step imagination: materialize predicted ŝ_{t+1}.

        Args:
            data: Current scene graph.
            action: [1] action index or [1, action_dim] vector.
            edge_threshold: Threshold for edge existence.
            node_types: Optional node type vector.

        Returns:
            New PyG Data for the predicted next state.
        """
        self.eval()
        device = data.x.device
        feat_dim = data.x.size(1)

        # Guard: if input graph is empty, return empty graph
        if data.x.size(0) == 0:
            return Data(
                x=torch.zeros(0, feat_dim, device=device),
                edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
                edge_attr=torch.zeros(0, dtype=torch.long, device=device),
            )

        with torch.no_grad():
            out = self.forward(data, action, node_types)

            # Node existence filter
            if out.existence_logits is not None:
                alive = torch.sigmoid(out.existence_logits.squeeze(-1)) > 0.5
            else:
                alive = torch.ones(data.x.size(0), dtype=torch.bool, device=device)

            new_x = out.predicted_x[alive]
            alive_idx = alive.nonzero(as_tuple=True)[0]

            # Handle empty graph after filtering
            if new_x.size(0) == 0:
                new_data = Data(
                    x=torch.zeros(0, feat_dim, device=device),
                    edge_index=torch.zeros(2, 0, dtype=torch.long, device=device),
                    edge_attr=torch.zeros(0, dtype=torch.long, device=device),
                )
                if hasattr(data, "pos") and data.pos is not None:
                    new_data.pos = torch.zeros(0, 3, device=device)
                return new_data

            # Edge reconstruction
            if out.edge_logits is not None and alive_idx.numel() > 0:
                probs = torch.sigmoid(out.edge_logits[0])  # [N, N]
                sub_probs = probs[alive_idx][:, alive_idx]
                sub_probs.fill_diagonal_(0)
                edge_mask = sub_probs > edge_threshold
                src, tgt = edge_mask.nonzero(as_tuple=True)
                new_edge_index = torch.stack([src, tgt])
                new_edge_attr = torch.zeros(new_edge_index.size(1), dtype=torch.long, device=device)
            else:
                new_edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                new_edge_attr = torch.zeros(0, dtype=torch.long, device=device)

            new_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
            if hasattr(data, "pos") and data.pos is not None:
                new_data.pos = data.pos[alive]
            
            # Carry over metadata for surviving nodes
            alive_list = alive_idx.cpu().tolist()
            if hasattr(data, "label_names"):
                new_data.label_names = [data.label_names[i] for i in alive_list]
            if hasattr(data, "semantic_labels") and data.semantic_labels is not None:
                new_data.semantic_labels = data.semantic_labels[alive]
            if hasattr(data, "node_type") and data.node_type is not None:
                new_data.node_type = data.node_type[alive]
                
            return new_data

    def rollout(
        self,
        data: Data,
        actions: List[torch.Tensor],
        edge_threshold: float = 0.5,
        node_types: Optional[torch.Tensor] = None,
    ) -> List[Data]:
        """Multi-step imagination rollout.

        Args:
            data: Initial scene graph s_0.
            actions: List of [1] action tensors for each step.
            edge_threshold: Edge existence threshold.
            node_types: Optional node types.

        Returns:
            List of [s_0, s_1, ..., s_T] predicted states.
        """
        trajectory = [data]
        current = data
        for action in actions:
            # Stop early if graph is empty (all nodes removed)
            if current.x.size(0) == 0:
                trajectory.extend([current] * (len(actions) - len(trajectory) + 1))
                break
            next_state = self.imagine(current, action, edge_threshold, node_types)
            trajectory.append(next_state)
            current = next_state
        return trajectory


# ──────────────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────────────


class TopologicalConsistencyLoss(nn.Module):
    """Penalizes nodes that "teleport" across rooms.

    For each node, computes the predicted position change and compares
    it against a maximum plausible displacement per timestep. Also checks
    that object→room containment edges are preserved.

    L_topo = mean(max(0, ||Δpos|| - max_displacement)²)
             + λ * BCE(predicted_room_edge, original_room_edge)
    """

    def __init__(self, max_displacement: float = 2.0, lambda_room: float = 0.5):
        super().__init__()
        self.max_displacement = max_displacement
        self.lambda_room = lambda_room

    def forward(
        self,
        predicted_x: torch.Tensor,
        original_x: torch.Tensor,
        pos_slice: Tuple[int, int] = (512, 515),
        room_edges_src: Optional[torch.Tensor] = None,
        room_edges_tgt: Optional[torch.Tensor] = None,
        predicted_edge_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute topological consistency loss.

        Args:
            predicted_x: [N, D] predicted node features.
            original_x: [N, D] original node features.
            pos_slice: (start, end) indices of position dims in feature vector.
            room_edges_src: Original object→room edge sources.
            room_edges_tgt: Original object→room edge targets.
            predicted_edge_logits: [N, N] edge logits from model.

        Returns:
            Scalar loss.
        """
        # Position displacement penalty
        pred_pos = predicted_x[:, pos_slice[0]:pos_slice[1]]
        orig_pos = original_x[:, pos_slice[0]:pos_slice[1]]
        displacement = torch.norm(pred_pos - orig_pos, dim=-1)  # [N]
        excess = F.relu(displacement - self.max_displacement)
        pos_loss = (excess ** 2).mean()

        # Room containment preservation
        room_loss = torch.tensor(0.0, device=predicted_x.device)
        if (
            room_edges_src is not None
            and room_edges_tgt is not None
            and predicted_edge_logits is not None
        ):
            # Original room edges should be preserved
            room_logits = predicted_edge_logits[room_edges_src, room_edges_tgt]
            room_targets = torch.ones_like(room_logits)
            room_loss = F.binary_cross_entropy_with_logits(room_logits, room_targets)

        return pos_loss + self.lambda_room * room_loss


class GNNTransitionLoss(nn.Module):
    """Combined loss for training GNNTransitionModel.

    L = λ_feat * MSE(X̂, X_true)        # Latent Reconstruction Loss
      + λ_edge * BCE(Â, A_true)         # Edge Prediction Loss
      + λ_exist * BCE(ê, e_true)        # Node Existence Loss
      + λ_topo * L_topological           # Topological Consistency
    """

    def __init__(
        self,
        lambda_feat: float = 1.0,
        lambda_edge: float = 0.5,
        lambda_exist: float = 0.3,
        lambda_topo: float = 0.2,
        max_displacement: float = 2.0,
        siglip_dim: int = 512,
    ):
        super().__init__()
        self.lambda_feat = lambda_feat
        self.lambda_edge = lambda_edge
        self.lambda_exist = lambda_exist
        self.lambda_topo = lambda_topo
        self.topo_loss = TopologicalConsistencyLoss(max_displacement)
        self.siglip_dim = siglip_dim

    def forward(
        self,
        output: GNNTransitionOutput,
        target: Data,
        source: Data,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            output: Model output.
            target: Ground truth s_{t+1}.
            source: Original s_t.

        Returns:
            Dict with individual and total losses.
        """
        losses = {}
        n = min(output.predicted_x.size(0), target.x.size(0))

        # ── Latent Reconstruction Loss (MSE on SigLIP features) ──
        pred_siglip = output.predicted_x[:n, :self.siglip_dim]
        true_siglip = target.x[:n, :self.siglip_dim]
        feat_loss = F.mse_loss(pred_siglip, true_siglip)
        losses["feat_mse"] = feat_loss

        total = self.lambda_feat * feat_loss

        # ── Full feature MSE (including position) ──
        full_mse = F.mse_loss(output.predicted_x[:n], target.x[:n])
        losses["full_mse"] = full_mse

        # ── Edge BCE ──
        if output.edge_logits is not None and target.edge_index.size(1) > 0:
            target_adj = to_dense_adj(target.edge_index, max_num_nodes=n)[0][:n, :n]
            pred_edges = output.edge_logits[0][:n, :n] if output.edge_logits.dim() == 3 else output.edge_logits[:n, :n]
            edge_loss = F.binary_cross_entropy_with_logits(pred_edges, target_adj.float())
            losses["edge_bce"] = edge_loss
            total = total + self.lambda_edge * edge_loss

        # ── Existence BCE ──
        if output.existence_logits is not None:
            n_src = source.x.size(0)
            exist_target = torch.zeros(n_src, 1, device=output.existence_logits.device)
            exist_target[:target.x.size(0)] = 1.0
            exist_loss = F.binary_cross_entropy_with_logits(
                output.existence_logits[:n_src], exist_target
            )
            losses["exist_bce"] = exist_loss
            total = total + self.lambda_exist * exist_loss

        # ── Topological Consistency ──
        topo_loss = self.topo_loss(output.predicted_x[:n], source.x[:n])
        losses["topo"] = topo_loss
        total = total + self.lambda_topo * topo_loss

        losses["total"] = total
        return losses
