"""Node feature encoder: combines DINOv2 visual features, 3D positional
encoding, semantic label embeddings, and layer indicators into a unified
per-node feature vector.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from worldgraph.config import NodeEncoderConfig
from worldgraph.scene_graph.graph_types import SceneGraph, SceneNode


class SinusoidalPositionalEncoding3D(nn.Module):
    """Sinusoidal positional encoding for 3D coordinates (x, y, z).

    Produces a fixed-size embedding for any 3D position, similar to
    the original Transformer positional encoding but extended to 3 axes.
    """

    def __init__(self, dim: int = 128):
        super().__init__()
        assert dim % 6 == 0 or dim % 3 == 0, "dim should be divisible by 3 or 6"
        self.dim = dim
        self.dim_per_axis = dim // 3

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode 3D positions.

        Args:
            positions: [N, 3] tensor of (x, y, z) coordinates.

        Returns:
            [N, dim] positional encodings.
        """
        device = positions.device
        encodings = []

        for axis in range(3):
            pos = positions[:, axis].unsqueeze(1)  # [N, 1]
            div_term = torch.exp(
                torch.arange(0, self.dim_per_axis, 2, device=device, dtype=torch.float32)
                * -(math.log(10000.0) / self.dim_per_axis)
            )  # [dim_per_axis // 2]

            enc = torch.zeros(pos.shape[0], self.dim_per_axis, device=device)
            enc[:, 0::2] = torch.sin(pos * div_term)
            enc[:, 1::2] = torch.cos(pos * div_term)
            encodings.append(enc)

        return torch.cat(encodings, dim=-1)  # [N, dim]


class BoundingBoxEncoder(nn.Module):
    """Encodes normalized bounding box features [x, y, z, w, h, d]."""

    def __init__(self, input_dim: int = 6, output_dim: int = 12):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        """Encode bounding boxes.

        Args:
            bboxes: [N, 6] tensor of [center_x, center_y, center_z, width, height, depth].

        Returns:
            [N, output_dim] bounding box features.
        """
        return self.mlp(bboxes)


class DINOv2FeatureExtractor(nn.Module):
    """Extracts per-object visual features using DINOv2 (frozen).

    Processes object crops through a pretrained DINOv2 ViT-L model
    to produce rich visual embeddings.
    """

    def __init__(self, model_name: str = "facebook/dinov2-vitl14", crop_size: int = 224):
        super().__init__()
        self.crop_size = crop_size
        self.model_name = model_name
        self._model = None
        self._transform = None

    def _lazy_load(self):
        """Lazy load model to avoid import-time GPU allocation."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoImageProcessor

        self._processor = AutoImageProcessor.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

    def to(self, device):
        """Move to device, loading model if needed."""
        super().to(device)
        self._lazy_load()
        self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def forward(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Extract DINOv2 features from object crops.

        Args:
            crops: List of [H, W, 3] uint8 numpy arrays (RGB object crops).

        Returns:
            [len(crops), feat_dim] tensor of visual features.
        """
        self._lazy_load()

        if len(crops) == 0:
            return torch.zeros(0, 1024, device=next(self._model.parameters()).device)

        # Convert numpy crops to PIL images
        pil_images = [Image.fromarray(crop) for crop in crops]

        # Process through DINOv2
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(next(self._model.parameters()).device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        # Use CLS token as the feature
        features = outputs.last_hidden_state[:, 0, :]  # [B, feat_dim]

        return features

    @torch.no_grad()
    def extract_single(self, crop: np.ndarray) -> torch.Tensor:
        """Extract feature for a single crop."""
        return self.forward([crop])[0]


class NodeFeatureEncoder(nn.Module):
    """Full node feature encoder combining all modalities.

    Produces a [N, total_feat_dim] tensor for all nodes in a scene graph,
    combining:
    - DINOv2 visual features (1024-d)
    - Sinusoidal 3D positional encoding (128-d)
    - Semantic label embedding (48-d)
    - Bounding box features (12-d)
    - Layer indicator (4-d)

    Total: 1216-d per node.
    """

    def __init__(self, config: NodeEncoderConfig):
        super().__init__()
        self.config = config

        # Visual encoder (frozen DINOv2)
        self.visual_encoder = DINOv2FeatureExtractor(
            model_name=config.visual_model,
            crop_size=config.visual_crop_size,
        )

        # Spatial encoding
        self.pos_encoder = SinusoidalPositionalEncoding3D(dim=config.position_encoding_dim)
        self.bbox_encoder = BoundingBoxEncoder(input_dim=6, output_dim=config.bbox_feat_dim)

        # Semantic encoding
        self.label_embedding = nn.Embedding(
            config.num_semantic_labels, config.semantic_emb_dim
        )

        # Layer encoding (one-hot, no parameters)
        self.num_layers = config.num_layers

    @property
    def output_dim(self) -> int:
        return self.config.total_feat_dim

    def forward(
        self,
        scene_graph: SceneGraph,
        precomputed_visual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode all nodes in a scene graph.

        Args:
            scene_graph: The scene graph to encode.
            precomputed_visual: Optional [N, visual_dim] tensor of precomputed
                DINOv2 features. If None, extracts from visual_crop on each node.

        Returns:
            [N, total_feat_dim] tensor of node features.
        """
        device = self.label_embedding.weight.device
        n = scene_graph.num_nodes

        # 1. Visual features
        if precomputed_visual is not None:
            visual_feats = precomputed_visual.to(device)
        else:
            crops = []
            for node in scene_graph.nodes:
                if node.visual_feature is not None:
                    crops.append(None)  # will use precomputed
                elif node.visual_crop is not None:
                    crops.append(node.visual_crop)
                else:
                    crops.append(None)

            # Separate precomputed vs needs-extraction
            needs_extraction = [i for i, c in enumerate(crops) if c is not None]
            visual_feats = torch.zeros(n, self.config.visual_feat_dim, device=device)

            if needs_extraction:
                extracted = self.visual_encoder([crops[i] for i in needs_extraction])
                for batch_idx, node_idx in enumerate(needs_extraction):
                    visual_feats[node_idx] = extracted[batch_idx]

            # Fill in precomputed features
            for i, node in enumerate(scene_graph.nodes):
                if node.visual_feature is not None:
                    visual_feats[i] = torch.tensor(
                        node.visual_feature, dtype=torch.float32, device=device
                    )

        # 2. Positional encoding
        positions = torch.tensor(
            [n.position for n in scene_graph.nodes],
            dtype=torch.float32, device=device,
        )
        pos_feats = self.pos_encoder(positions)  # [N, 128]

        # 3. Bounding box features
        bboxes = torch.tensor(
            [n.bounding_box[:6] for n in scene_graph.nodes],
            dtype=torch.float32, device=device,
        )
        bbox_feats = self.bbox_encoder(bboxes)  # [N, 12]

        # 4. Semantic label embedding
        labels = torch.tensor(
            [n.semantic_label for n in scene_graph.nodes],
            dtype=torch.long, device=device,
        )
        label_feats = self.label_embedding(labels)  # [N, 48]

        # 5. Layer indicator (one-hot)
        layer_ids = torch.tensor(
            [n.layer.value for n in scene_graph.nodes],
            dtype=torch.long, device=device,
        )
        layer_feats = torch.zeros(n, self.num_layers, device=device)
        layer_feats.scatter_(1, layer_ids.unsqueeze(1), 1.0)  # [N, 4]

        # Concatenate all features
        node_features = torch.cat(
            [visual_feats, pos_feats, bbox_feats, label_feats, layer_feats],
            dim=-1,
        )  # [N, 1216]

        return node_features
