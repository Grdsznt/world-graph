"""Training pipeline for the WorldGraph transition model.

Trains on (s_t, a_t, s_{t+1}) triples from simulated environments.
Only the transition model parameters are trained; CLIP and DINOv2 remain frozen.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from worldgraph.config import WorldGraphConfig, TrainingConfig
from worldgraph.world_model.transition import TransitionModel, TransitionOutput
from worldgraph.world_model.action_encoder import ActionEncoder


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class TransitionLoss(nn.Module):
    """Combined loss for training the transition model.

    L = λ_node * L_node + λ_edge * L_edge + λ_exist * L_exist

    Where:
    - L_node: MSE between predicted and true node features
    - L_edge: BCE between predicted and true edge adjacency
    - L_exist: BCE between predicted and true node existence
    """

    def __init__(
        self,
        lambda_node: float = 1.0,
        lambda_edge: float = 0.5,
        lambda_exist: float = 0.3,
    ):
        super().__init__()
        self.lambda_node = lambda_node
        self.lambda_edge = lambda_edge
        self.lambda_exist = lambda_exist

    def forward(
        self,
        output: TransitionOutput,
        target_graph: Data,
        source_graph: Data,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            output: TransitionOutput from the transition model.
            target_graph: Ground-truth next-state graph s_{t+1}.
            source_graph: Current-state graph s_t (for node alignment).

        Returns:
            Dict with 'total', 'node', 'edge', 'exist' losses.
        """
        losses = {}

        # Node feature prediction loss (MSE)
        # Only compute for nodes that exist in both s_t and s_{t+1}
        n_source = source_graph.x.size(0)
        n_target = target_graph.x.size(0)
        n_common = min(n_source, n_target)

        node_loss = F.mse_loss(
            output.predicted_node_features[:n_common],
            target_graph.x[:n_common],
        )
        losses["node"] = node_loss

        total_loss = self.lambda_node * node_loss

        # Edge prediction loss (BCE)
        if output.edge_logits is not None:
            # Build ground-truth dense adjacency for target graph
            if target_graph.edge_index.size(1) > 0:
                target_adj = to_dense_adj(
                    target_graph.edge_index,
                    max_num_nodes=n_common,
                )[0][:n_common, :n_common]
            else:
                target_adj = torch.zeros(
                    n_common, n_common,
                    device=output.edge_logits.device,
                )

            # Get predicted edge logits for the common nodes
            pred_edge = output.edge_logits
            if pred_edge.dim() == 3:
                pred_edge = pred_edge[0]  # take first graph in batch
            pred_edge = pred_edge[:n_common, :n_common]

            edge_loss = F.binary_cross_entropy_with_logits(
                pred_edge, target_adj.float()
            )
            losses["edge"] = edge_loss
            total_loss = total_loss + self.lambda_edge * edge_loss

        # Node existence loss (BCE)
        if output.node_existence_logits is not None:
            # Ground truth: 1.0 for nodes that exist in target, 0.0 otherwise
            exist_target = torch.zeros(
                n_source, 1,
                device=output.node_existence_logits.device,
            )
            exist_target[:n_target] = 1.0

            exist_loss = F.binary_cross_entropy_with_logits(
                output.node_existence_logits[:n_source],
                exist_target,
            )
            losses["exist"] = exist_loss
            total_loss = total_loss + self.lambda_exist * exist_loss

        losses["total"] = total_loss
        return losses


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TransitionDataset(Dataset):
    """Dataset of (s_t, a_t, s_{t+1}) transition triples.

    Each sample contains:
    - source_graph: PyG Data for s_t
    - action_text: string action description
    - target_graph: PyG Data for s_{t+1}
    """

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing transition data files.
                Expected structure:
                    data_dir/
                        transitions.json  # metadata + action text
                        graphs/           # .pt files for each graph state
        """
        self.data_dir = Path(data_dir)
        self.transitions = []

        # Load transition metadata
        meta_path = self.data_dir / "transitions.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.transitions = json.load(f)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict:
        entry = self.transitions[idx]

        # Load source and target graphs
        source_path = self.data_dir / "graphs" / f"{entry['source_id']}.pt"
        target_path = self.data_dir / "graphs" / f"{entry['target_id']}.pt"

        source_graph = torch.load(source_path, weights_only=False)
        target_graph = torch.load(target_path, weights_only=False)

        return {
            "source_graph": source_graph,
            "action_text": entry["action"],
            "target_graph": target_graph,
        }


def collate_transitions(batch: List[Dict]) -> Dict:
    """Custom collate function for transition triples."""
    source_graphs = Batch.from_data_list([b["source_graph"] for b in batch])
    target_graphs = Batch.from_data_list([b["target_graph"] for b in batch])
    action_texts = [b["action_text"] for b in batch]

    return {
        "source_graph": source_graphs,
        "action_texts": action_texts,
        "target_graph": target_graphs,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class WorldGraphTrainer:
    """Training loop for the transition model."""

    def __init__(
        self,
        transition_model: TransitionModel,
        action_encoder: ActionEncoder,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = transition_model.to(device)
        self.action_enc = action_encoder.to(device)
        self.config = config
        self.device = device

        # Loss
        self.criterion = TransitionLoss(
            lambda_node=config.lambda_node,
            lambda_edge=config.lambda_edge,
            lambda_exist=config.lambda_exist,
        )

        # Optimizer — only transition model params (CLIP is frozen)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler
        self.scheduler = None  # set in train() after knowing total steps

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if config.mixed_precision == "fp16" else None
        self.amp_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    def train(
        self,
        train_dataset: TransitionDataset,
        val_dataset: Optional[TransitionDataset] = None,
    ):
        """Run the full training loop.

        Args:
            train_dataset: Training data.
            val_dataset: Optional validation data.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_transitions,
            num_workers=4,
            pin_memory=True,
        )

        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)

        # Cosine LR scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=self.config.warmup_ratio,
            anneal_strategy="cos",
        )

        # Training loop
        os.makedirs(self.config.output_dir, exist_ok=True)
        global_step = 0

        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_losses = {"total": 0.0, "node": 0.0, "edge": 0.0, "exist": 0.0}
            epoch_start = time.time()

            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                source = batch["source_graph"].to(self.device)
                target = batch["target_graph"].to(self.device)
                action_texts = batch["action_texts"]

                # Encode actions (frozen CLIP, no grad)
                with torch.no_grad():
                    action_embs = self.action_enc(action_texts)

                # Forward pass with mixed precision
                with torch.amp.autocast(
                    "cuda",
                    dtype=self.amp_dtype,
                    enabled=self.config.mixed_precision != "none",
                ):
                    output = self.model(source, action_embs)
                    losses = self.criterion(output, target, source)

                # Backward pass
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(losses["total"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.scheduler.step()
                global_step += 1

                # Accumulate losses
                for k, v in losses.items():
                    epoch_losses[k] += v.item()

                # Logging
                if global_step % self.config.log_every_n_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(
                        f"[Step {global_step}] "
                        f"loss={losses['total'].item():.4f} "
                        f"node={losses.get('node', torch.tensor(0)).item():.4f} "
                        f"edge={losses.get('edge', torch.tensor(0)).item():.4f} "
                        f"exist={losses.get('exist', torch.tensor(0)).item():.4f} "
                        f"lr={lr:.2e}"
                    )

            # Epoch summary
            n_batches = len(train_loader)
            elapsed = time.time() - epoch_start
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
            print(
                f"\n[Epoch {epoch + 1}/{self.config.num_epochs}] "
                f"avg_loss={avg_losses['total']:.4f} "
                f"time={elapsed:.1f}s"
            )

            # Validation
            if val_dataset is not None:
                val_loss = self.validate(val_dataset)
                print(f"  val_loss={val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1)

        # Save final checkpoint
        self.save_checkpoint("final")
        print(f"\nTraining complete. Checkpoints saved to {self.config.output_dir}")

    @torch.no_grad()
    def validate(self, val_dataset: TransitionDataset) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_transitions,
            num_workers=4,
        )

        total_loss = 0.0
        for batch in val_loader:
            source = batch["source_graph"].to(self.device)
            target = batch["target_graph"].to(self.device)
            action_embs = self.action_enc(batch["action_texts"])

            output = self.model(source, action_embs)
            losses = self.criterion(output, target, source)
            total_loss += losses["total"].item()

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        path = os.path.join(self.config.output_dir, f"checkpoint_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
