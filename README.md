# WorldGraph: 3D Scene Graph World Model

WorldGraph is a Graph Neural Network (GNN) based World Model designed for Embodied AI and AR/XR applications. It learns the "Physics and Rules of the World" by predicting how 3D Scene Graphs transition when high-level actions are taken ($s_t \xrightarrow{a_t} s_{t+1}$).

This repository is optimized to run on consumer hardware (e.g., RTX 3090, 24GB VRAM) and is designed to integrate with the [Hydra 3D Scene Graph](https://github.com/MIT-SPARK/Hydra) pipeline.

---

## 🚀 Quick Start & Installation

### 1. Environment Setup
It is highly recommended to use Conda. The model requires PyTorch and PyTorch Geometric.

```bash
# Create and activate environment
conda create -n worldgraph python=3.11 -y
conda activate worldgraph

# Install PyTorch (Adjust CUDA version for your machine, here assuming CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch_geometric

# Install other dependencies
pip install -r requirements.txt
```

### 2. Verify Installation
Run the test suites to ensure the architecture and tensor shapes are correct for your hardware.

```bash
# Run the spec-compliant GNN Transition tests (Hydra JSON parsing, Rollout, Identity)
python tests/test_gnn_transition.py

# Run the GPS Transformer shape tests
python tests/test_shapes.py
```

### 3. Run the Training Demo
To see the model learn deterministic dynamics (pick up, put down, search) on synthetic kitchen graphs:

```bash
# On a CUDA machine (e.g., RTX 3090)
CUDA_VISIBLE_DEVICES=0 python scripts/demo_train.py

# On Apple Silicon (MPS) or CPU
python scripts/demo_train.py
```

---

## 🧠 Architecture Variants

This repository contains two variants of the Transition Model $T(s_t, a_t) \to s_{t+1}$:

1. **`GNNTransitionModel` (Survey Compliant)** - `world_model/gnn_transition.py`
   * Uses GATv2 for local message passing.
   * Action injection via **FiLM** (Feature-wise Linear Modulation).
   * Implements **Topological Consistency Loss**.
   * Best for matching academic surveys and fast ablation studies (~2.5M params).

2. **`TransitionModel` (GPS)** - `world_model/transition.py`
   * Uses GPS blocks (GATv2 + Global Transformer Self-Attention).
   * Action injection via Cross-Attention.
   * Scaled up to ~150M params for complex global scene reasoning.

---

## 🛠️ Next Steps for Integration

To move from this synthetic proof-of-concept to a real-world AR/XR application, follow these phases:

### Phase 1: Real-World Data Integration (Hydra)
Currently, `scene_graph/hydra_json_parser.py` parses Hydra JSONs into PyG Data objects.
1. Run your Meta Quest 3 data through the Hydra pipeline to generate real `.json` scene graphs.
2. Verify the parser handles your specific Hydra output format.
3. Replace the synthetic graphs in `demo_train.py` with real pairs of $(G_t, G_{t+1})$ from your Hydra logs.

### Phase 2: Open-Vocabulary Features (SigLIP / DINOv2)
1. Hook up the `PerceptionPipeline` (`perception/detector.py`).
2. Crop the bounding boxes from your real RGB data and pass them through SigLIP/DINOv2.
3. Inject these 512-d feature vectors into the `features` array in the Hydra JSON before passing it to the WorldGraph parser.

### Phase 3: Simulator Dataset Generation
To train the model to understand actions, you need a large dataset.
1. Set up **AI2-THOR** or **ProcTHOR** in headless mode on your 3090 server.
2. Write an exploration script:
   ```python
   # Pseudocode
   G_t = hydra_pipeline(thor_rgbd_t)
   action = random_action()
   thor.step(action)
   G_t1 = hydra_pipeline(thor_rgbd_t1)
   save(G_t, action, G_t1)
   ```
3. Train the model using `scripts/train.py` on this massive dataset.

### Phase 4: The Imagination Engine
Once trained, deploy the planning module:
1. When a user asks "Where is my mug?", encode "mug" into a goal vector using SigLIP.
2. Use `planning/imagination.py` to `rollout()` possible actions.
3. The `LostObjectScorer` (`planning/scoring.py`) will check the imagined graphs against the goal vector.
4. Pass the winning action into the `RecommendationGenerator` (`output/recommender.py`) to stream a text tip to the AR headset via the LLaMA 3.1 4-bit model.

---

## 📁 Repository Structure

```text
worldgraph/
├── worldgraph/
│   ├── config.py                 # Hyperparameters and settings
│   ├── scene_graph/              # Graph structures and Hydra integration
│   ├── world_model/              # GNN Transition models and encoders
│   ├── planning/                 # Imagination engine and heuristic scorers
│   ├── perception/               # YOLOE + SAM2 bindings
│   └── output/                   # LLM UI recommendation generation
├── scripts/
│   ├── train.py                  # Full dataset training loop
│   └── demo_train.py             # Synthetic deterministic training demo
└── tests/                        # Validation suites
```
