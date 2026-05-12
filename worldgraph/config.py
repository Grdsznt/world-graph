"""Configuration dataclasses for WorldGraph."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NodeEncoderConfig:
    """Configuration for node feature encoding."""
    # DINOv2 visual features
    visual_model: str = "facebook/dinov2-vitl14"
    visual_feat_dim: int = 1024  # DINOv2 ViT-L output dim
    visual_crop_size: int = 224

    # Spatial encoding
    position_encoding_dim: int = 128  # sinusoidal 3D pos encoding
    bbox_feat_dim: int = 12  # normalized [x,y,z,w,h,d] + extras

    # Semantic encoding
    num_semantic_labels: int = 200  # max label vocabulary
    semantic_emb_dim: int = 48

    # Layer indicator (object / place / room / building)
    num_layers: int = 4
    layer_emb_dim: int = 4

    @property
    def total_feat_dim(self) -> int:
        return (
            self.visual_feat_dim
            + self.position_encoding_dim
            + self.bbox_feat_dim
            + self.semantic_emb_dim
            + self.layer_emb_dim
        )


@dataclass
class GraphEncoderConfig:
    """Configuration for GPS Graph Transformer encoder."""
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    # GATv2 local attention config
    gat_heads: int = 8
    # Edge type encoding
    num_edge_types: int = 12  # isOn, isIn, nearBy, supports, etc.
    edge_emb_dim: int = 64


@dataclass
class TransitionModelConfig:
    """Configuration for the world model transition function T(s_t, a_t) → s_{t+1}."""
    graph_encoder: GraphEncoderConfig = field(default_factory=GraphEncoderConfig)
    node_encoder: NodeEncoderConfig = field(default_factory=NodeEncoderConfig)

    # Action encoding (CLIP text)
    action_encoder_model: str = "openai/clip-vit-large-patch14"
    action_feat_dim: int = 768  # CLIP ViT-L text output dim

    # Cross-attention for action conditioning
    cross_attn_heads: int = 8

    # Prediction head configs
    predict_node_existence: bool = True
    predict_edge_changes: bool = True


@dataclass
class ImaginationConfig:
    """Configuration for imagination / planning."""
    max_rollout_depth: int = 1  # single-step default, extendable
    num_action_candidates: int = 10  # evaluate top-N actions
    edge_threshold: float = 0.5  # threshold for edge existence in predicted graph


@dataclass
class PerceptionConfig:
    """Configuration for the perception pipeline."""
    # YOLOE
    yoloe_model: str = "yoloe-11l-seg.pt"
    yoloe_conf_threshold: float = 0.3

    # SAM2 (backup refinement)
    use_sam2_refinement: bool = False
    sam2_model: str = "facebook/sam2-hiera-tiny"

    # DINOv2 (per-object features)
    dinov2_model: str = "facebook/dinov2-vitl14"


@dataclass
class OutputConfig:
    """Configuration for recommendation output generation."""
    llm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm_quantization: str = "4bit"
    max_new_tokens: int = 128
    temperature: float = 0.7


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 50
    warmup_ratio: float = 0.05
    lr_scheduler: str = "cosine"

    # Loss weights
    lambda_node: float = 1.0  # node feature MSE
    lambda_edge: float = 0.5  # edge BCE
    lambda_exist: float = 0.3  # node existence BCE

    # Gradient checkpointing (saves VRAM)
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"

    # Logging
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 5
    output_dir: str = "./checkpoints"


@dataclass
class WorldGraphConfig:
    """Top-level configuration."""
    transition: TransitionModelConfig = field(default_factory=TransitionModelConfig)
    imagination: ImaginationConfig = field(default_factory=ImaginationConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    device: str = "cuda"
    seed: int = 42
