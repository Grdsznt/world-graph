"""Demo: Train and test the GNN Transition Model on synthetic kitchen data.

This script generates (s_t, a_t, s_{t+1}) triples with known, deterministic
dynamics, trains the model to learn them, and verifies it produces meaningful
predictions.

Run on the 3090:
    /opt/anaconda3/envs/ml/bin/python scripts/demo_train.py

Or with CUDA specifically:
    CUDA_VISIBLE_DEVICES=0 /opt/anaconda3/envs/ml/bin/python scripts/demo_train.py

Scenario:
    Kitchen with objects on surfaces. Actions have deterministic effects:
    - Action 0 (null):     Nothing changes
    - Action 1 (pick_up):  Target object position moves to "hand" position
    - Action 2 (put_down): Object moves from hand to target surface
    - Action 3 (open):     Container state changes (drawer, fridge)
    - Action 4 (search):   Hidden objects become visible (position revealed)
"""

import sys
import os
import time
import json

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldgraph.world_model.gnn_transition import (
    GNNTransitionModel,
    GNNTransitionConfig,
    GNNTransitionLoss,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic Data Generation
# ──────────────────────────────────────────────────────────────────────


# Semantic "fingerprint" embeddings — each object has a distinct feature pattern
# In practice these would be SigLIP embeddings; here we use hand-crafted
# orthogonal-ish vectors so we can verify semantic preservation.
OBJECT_EMBEDDINGS = {
    "mug":     torch.randn(512) * 0.1 + torch.eye(512)[0] * 2.0,
    "plate":   torch.randn(512) * 0.1 + torch.eye(512)[50] * 2.0,
    "fork":    torch.randn(512) * 0.1 + torch.eye(512)[100] * 2.0,
    "knife":   torch.randn(512) * 0.1 + torch.eye(512)[150] * 2.0,
    "bottle":  torch.randn(512) * 0.1 + torch.eye(512)[200] * 2.0,
    "keys":    torch.randn(512) * 0.1 + torch.eye(512)[250] * 2.0,
    "counter": torch.randn(512) * 0.1 + torch.eye(512)[300] * 2.0,
    "sink":    torch.randn(512) * 0.1 + torch.eye(512)[350] * 2.0,
    "table":   torch.randn(512) * 0.1 + torch.eye(512)[400] * 2.0,
}

# Fixed positions for surfaces
SURFACE_POSITIONS = {
    "counter": torch.tensor([0.0, 0.0, 0.8]),
    "sink":    torch.tensor([1.5, 0.0, 0.8]),
    "table":   torch.tensor([0.0, 2.0, 0.7]),
}

# Hand position (where objects go when picked up)
HAND_POS = torch.tensor([0.5, 0.5, 1.2])


def make_kitchen_graph(
    objects_on_counter=("mug", "plate", "fork"),
    objects_on_table=("knife",),
    objects_in_hand=(),
    objects_hidden=("keys",),
    seed=None,
):
    """Create a kitchen scene graph with known object placements.

    Returns PyG Data with x=[N, 515], edge_index, edge_attr, etc.
    """
    if seed is not None:
        torch.manual_seed(seed)

    nodes = []
    positions = []
    names = []
    node_types = []  # 0=object, 1=place

    # Add surfaces first
    for surface_name in ["counter", "sink", "table"]:
        emb = OBJECT_EMBEDDINGS[surface_name].clone()
        pos = SURFACE_POSITIONS[surface_name].clone()
        nodes.append(torch.cat([emb, pos]))
        positions.append(pos)
        names.append(surface_name)
        node_types.append(1)  # place/surface

    # Add objects
    for obj_name in objects_on_counter:
        emb = OBJECT_EMBEDDINGS[obj_name].clone()
        # Small random offset from counter position
        pos = SURFACE_POSITIONS["counter"].clone() + torch.randn(3) * 0.05
        nodes.append(torch.cat([emb, pos]))
        positions.append(pos)
        names.append(obj_name)
        node_types.append(0)

    for obj_name in objects_on_table:
        emb = OBJECT_EMBEDDINGS[obj_name].clone()
        pos = SURFACE_POSITIONS["table"].clone() + torch.randn(3) * 0.05
        nodes.append(torch.cat([emb, pos]))
        positions.append(pos)
        names.append(obj_name)
        node_types.append(0)

    for obj_name in objects_in_hand:
        emb = OBJECT_EMBEDDINGS[obj_name].clone()
        pos = HAND_POS.clone() + torch.randn(3) * 0.02
        nodes.append(torch.cat([emb, pos]))
        positions.append(pos)
        names.append(obj_name)
        node_types.append(0)

    for obj_name in objects_hidden:
        emb = OBJECT_EMBEDDINGS[obj_name].clone()
        # Hidden objects have a "hidden" position (far away / unknown)
        pos = torch.tensor([10.0, 10.0, 10.0])
        nodes.append(torch.cat([emb, pos]))
        positions.append(pos)
        names.append(obj_name)
        node_types.append(0)

    x = torch.stack(nodes)  # [N, 515]
    pos = torch.stack(positions)  # [N, 3]

    # Build edges: objects → nearest surface (is_on = 1)
    src, tgt, etypes = [], [], []
    surface_indices = [0, 1, 2]  # counter, sink, table

    for i in range(3, len(names)):
        # Find nearest surface
        obj_pos = pos[i]
        min_dist, nearest_surface = float("inf"), 0
        for s_idx in surface_indices:
            d = torch.norm(obj_pos - pos[s_idx]).item()
            if d < min_dist:
                min_dist = d
                nearest_surface = s_idx

        # Only add edge if close enough (not hidden)
        if min_dist < 3.0:
            # Bidirectional
            src.extend([i, nearest_surface])
            tgt.extend([nearest_surface, i])
            etypes.extend([1, 1])  # is_on

    # Add surface connectivity
    for i in range(len(surface_indices) - 1):
        src.extend([surface_indices[i], surface_indices[i + 1]])
        tgt.extend([surface_indices[i + 1], surface_indices[i]])
        etypes.extend([6, 6])  # connected

    if src:
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        edge_attr = torch.tensor(etypes, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    data.node_type = torch.tensor(node_types, dtype=torch.long)
    data.label_names = names
    data.num_nodes = x.size(0)
    return data


def generate_training_pairs(num_samples=200, seed=42):
    """Generate (s_t, action, s_{t+1}) triples with deterministic effects.

    Effects:
    - Action 0 (null):    s_{t+1} = s_t (identity)
    - Action 1 (pick_up): mug moves from counter to hand
    - Action 2 (put_down): mug moves from hand to table
    - Action 3 (search):  keys position moves from hidden to counter
    """
    torch.manual_seed(seed)
    pairs = []

    for i in range(num_samples):
        action_type = i % 4  # cycle through actions
        if action_type == 0:
            # NULL: nothing changes
            s_t = make_kitchen_graph(
                objects_on_counter=("mug", "plate", "fork"),
                objects_on_table=("knife",),
                objects_hidden=("keys",),
                seed=i,
            )
            s_t1 = make_kitchen_graph(
                objects_on_counter=("mug", "plate", "fork"),
                objects_on_table=("knife",),
                objects_hidden=("keys",),
                seed=i,  # same seed = same graph
            )
            action = 0

        elif action_type == 1:
            # PICK UP MUG: mug moves from counter to hand
            s_t = make_kitchen_graph(
                objects_on_counter=("mug", "plate", "fork"),
                objects_on_table=("knife",),
                objects_hidden=("keys",),
                seed=i,
            )
            s_t1 = make_kitchen_graph(
                objects_on_counter=("plate", "fork"),
                objects_on_table=("knife",),
                objects_in_hand=("mug",),
                objects_hidden=("keys",),
                seed=i + 1000,
            )
            action = 1

        elif action_type == 2:
            # PUT DOWN MUG ON TABLE: mug moves from hand to table
            s_t = make_kitchen_graph(
                objects_on_counter=("plate", "fork"),
                objects_on_table=("knife",),
                objects_in_hand=("mug",),
                objects_hidden=("keys",),
                seed=i,
            )
            s_t1 = make_kitchen_graph(
                objects_on_counter=("plate", "fork"),
                objects_on_table=("knife", "mug"),
                objects_hidden=("keys",),
                seed=i + 1000,
            )
            action = 2

        elif action_type == 3:
            # SEARCH: keys become visible on counter
            s_t = make_kitchen_graph(
                objects_on_counter=("mug", "plate"),
                objects_on_table=("knife",),
                objects_hidden=("keys",),
                seed=i,
            )
            s_t1 = make_kitchen_graph(
                objects_on_counter=("mug", "plate", "keys"),
                objects_on_table=("knife",),
                seed=i + 1000,
            )
            action = 3

        pairs.append((s_t, action, s_t1))

    return pairs


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────


def train(model, loss_fn, pairs, device, num_epochs=300, lr=1e-3):
    """Train the model on synthetic data."""
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"\n{'='*60}")
    print(f"Training on {len(pairs)} samples, {num_epochs} epochs")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    start = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_losses = {}

        # Shuffle pairs
        indices = torch.randperm(len(pairs))

        for idx in indices:
            s_t, action_idx, s_t1 = pairs[idx]
            s_t = s_t.to(device)
            s_t1 = s_t1.to(device)
            action = torch.tensor([action_idx], device=device)

            # Forward
            output = model(s_t, action, s_t.node_type)
            losses = loss_fn(output, s_t1, s_t)

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += losses["total"].item()
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()

        scheduler.step()
        avg_loss = epoch_loss / len(pairs)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr_now = scheduler.get_last_lr()[0]
            parts = " | ".join(f"{k}: {v/len(pairs):.4f}" for k, v in epoch_losses.items() if k != "total")
            print(f"Epoch {epoch+1:>3d}/{num_epochs} | loss: {avg_loss:.4f} | {parts} | lr: {lr_now:.1e}")

    elapsed = time.time() - start
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed/num_epochs*1000:.0f}ms/epoch)")
    return model


# ──────────────────────────────────────────────────────────────────────
# Evaluation / Demo
# ──────────────────────────────────────────────────────────────────────


def evaluate(model, pairs, device):
    """Evaluate model predictions and print readable results."""
    model.eval()
    model.to(device)

    action_names = {0: "null (do nothing)", 1: "pick_up mug", 2: "put_down mug→table", 3: "search for keys"}

    print(f"\n{'='*60}")
    print("EVALUATION — Predicted vs Ground Truth")
    print(f"{'='*60}\n")

    # Test one of each action type
    for action_type in range(4):
        # Find a sample with this action
        for s_t, a, s_t1 in pairs:
            if a == action_type:
                break

        s_t = s_t.to(device)
        s_t1 = s_t1.to(device)
        action = torch.tensor([action_type], device=device)

        with torch.no_grad():
            output = model(s_t, action, s_t.node_type)

        print(f"Action: {action_names[action_type]}")
        print(f"  {'Node':<12} {'s_t pos':>20} {'ŝ_{t+1} pos (pred)':>25} {'s_{t+1} pos (true)':>25} {'Δ norm':>8}")
        print(f"  {'─'*12} {'─'*20} {'─'*25} {'─'*25} {'─'*8}")

        n = min(s_t.x.size(0), s_t1.x.size(0), output.predicted_x.size(0))
        for i in range(n):
            name = s_t.label_names[i] if hasattr(s_t, "label_names") else f"node_{i}"
            pos_t = s_t.x[i, 512:515].cpu().numpy()
            pos_pred = output.predicted_x[i, 512:515].cpu().numpy()
            pos_true = s_t1.x[i, 512:515].cpu().numpy()
            delta = torch.norm(output.delta_x[i]).item()

            pos_t_str = f"({pos_t[0]:>5.2f}, {pos_t[1]:>5.2f}, {pos_t[2]:>5.2f})"
            pos_pred_str = f"({pos_pred[0]:>5.2f}, {pos_pred[1]:>5.2f}, {pos_pred[2]:>5.2f})"
            pos_true_str = f"({pos_true[0]:>5.2f}, {pos_true[1]:>5.2f}, {pos_true[2]:>5.2f})"

            print(f"  {name:<12} {pos_t_str:>20} {pos_pred_str:>25} {pos_true_str:>25} {delta:>8.4f}")

        # Feature-level accuracy (cosine similarity of predicted vs true SigLIP)
        pred_sig = output.predicted_x[:n, :512]
        true_sig = s_t1.x[:n, :512]
        cos_sims = F.cosine_similarity(pred_sig, true_sig, dim=-1)
        avg_cos = cos_sims.mean().item()

        # Position MSE
        pred_pos = output.predicted_x[:n, 512:515]
        true_pos = s_t1.x[:n, 512:515]
        pos_mse = F.mse_loss(pred_pos, true_pos).item()

        print(f"  → Avg cosine sim (SigLIP features): {avg_cos:.4f}")
        print(f"  → Position MSE: {pos_mse:.6f}")
        print()

    # Imagination rollout demo
    print(f"{'='*60}")
    print("IMAGINATION ROLLOUT — 3-step sequence")
    print(f"{'='*60}\n")

    # Scenario: pick up mug → put down on table → search for keys
    scenario_actions = [1, 2, 3]
    scenario_names = ["pick_up mug", "put_down mug→table", "search for keys"]

    start_graph = make_kitchen_graph(
        objects_on_counter=("mug", "plate", "fork"),
        objects_on_table=("knife",),
        objects_hidden=("keys",),
        seed=999,
    ).to(device)

    print("Starting scene:")
    for i, name in enumerate(start_graph.label_names):
        pos = start_graph.x[i, 512:515].cpu().numpy()
        print(f"  {name:<12} at ({pos[0]:>5.2f}, {pos[1]:>5.2f}, {pos[2]:>5.2f})")

    current = start_graph
    for step, (act, act_name) in enumerate(zip(scenario_actions, scenario_names)):
        action = torch.tensor([act], device=device)
        current = model.imagine(current, action, edge_threshold=0.3)

        print(f"\nAfter step {step+1} ({act_name}): {current.x.size(0)} nodes")
        if current.x.size(0) == 0:
            print("  ⚠ Graph became empty — model needs more training")
            break
        for i in range(current.x.size(0)):
            pos = current.x[i, 512:515].cpu().numpy()
            # Find closest known embedding to identify the node
            best_name, best_sim = "unknown", -1
            for obj_name, emb in OBJECT_EMBEDDINGS.items():
                sim = F.cosine_similarity(
                    current.x[i, :512].unsqueeze(0),
                    emb.unsqueeze(0).to(device),
                ).item()
                if sim > best_sim:
                    best_sim = sim
                    best_name = obj_name
            print(f"  {best_name:<12} at ({pos[0]:>5.2f}, {pos[1]:>5.2f}, {pos[2]:>5.2f})  [sim={best_sim:.3f}]")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main():
    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Device: CPU (no GPU found)")

    # Config — small model for fast demo
    config = GNNTransitionConfig(
        node_feat_dim=515,
        siglip_dim=512,
        pos_dim=3,
        hidden_dim=256,
        num_gat_layers=4,
        gat_heads=8,
        num_actions=10,
        action_dim=64,
        film_hidden_dim=128,
        predict_edges=True,
        predict_existence=False,  # disable for demo (keep all nodes)
    )

    model = GNNTransitionModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters ({total_params * 4 / 1024**2:.1f} MB fp32)")

    loss_fn = GNNTransitionLoss(
        lambda_feat=1.0,
        lambda_edge=0.3,
        lambda_exist=0.0,  # disabled
        lambda_topo=0.1,
    )

    # Generate training data
    print("\nGenerating synthetic kitchen training data...")
    pairs = generate_training_pairs(num_samples=200, seed=42)
    print(f"Generated {len(pairs)} (s_t, a_t, s_t+1) triples")
    print(f"  Graph sizes: {pairs[0][0].x.size(0)} nodes, {pairs[0][0].edge_index.size(1)} edges")

    # Train
    model = train(model, loss_fn, pairs, device, num_epochs=300, lr=1e-3)

    # Evaluate
    evaluate(model, pairs, device)

    # VRAM usage report
    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1024**2
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"\n{'='*60}")
        print(f"VRAM Report")
        print(f"  Peak usage:  {mem_used:.0f} MB")
        print(f"  Total VRAM:  {mem_total:.0f} MB")
        print(f"  Remaining:   {mem_total - mem_used:.0f} MB")
        print(f"  Utilization: {mem_used/mem_total*100:.1f}%")
        print(f"{'='*60}")

    # Save checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "demo_trained.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel saved to {ckpt_path}")


if __name__ == "__main__":
    main()
