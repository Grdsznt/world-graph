"""Demo: Evaluate the trained GNN Transition Model.

Loads a saved checkpoint and runs the evaluation and imagination rollout
on synthetic kitchen data.

Run:
    python scripts/demo_eval.py [--checkpoint checkpoints/demo_trained.pt]
"""

import sys
import os
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldgraph.world_model.gnn_transition import GNNTransitionModel, GNNTransitionConfig
# Reuse the synthetic data generators and evaluation logic from the train script
from scripts.demo_train import generate_training_pairs, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained GNN Transition Model")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "demo_trained.pt"),
        help="Path to model checkpoint"
    )
    args = parser.parse_args()

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("Please run `python scripts/demo_train.py` first to generate a checkpoint.")
        sys.exit(1)

    print(f"Loading checkpoint from: {args.checkpoint}")

    # Config must match the one used during demo_train.py
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
        predict_existence=False,  # disabled for demo
    )

    # Initialize model and load weights
    model = GNNTransitionModel(config)
    
    # Use weights_only=True for security best practices when loading pickles
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        
    model.to(device)
    model.eval()

    print("\nGenerating synthetic kitchen test data (seed=123)...")
    # Use a different seed from training (which used 42) to evaluate generalization
    test_pairs = generate_training_pairs(num_samples=50, seed=123)
    
    # Run the evaluation suite
    evaluate(model, test_pairs, device)


if __name__ == "__main__":
    main()
