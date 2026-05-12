"""Scene graph data structures compatible with PyTorch Geometric.

Defines the intermediate representation between Hydra's spark_dsg output
and the world model's PyG-based input.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from torch_geometric.data import Data, Batch


class NodeLayer(IntEnum):
    """Hydra scene graph hierarchy layers."""
    OBJECT = 0
    PLACE = 1
    ROOM = 2
    BUILDING = 3


class EdgeType(IntEnum):
    """Spatial/semantic edge types in the scene graph."""
    IS_ON = 0       # object is on surface
    IS_IN = 1       # object is inside container/room
    NEAR_BY = 2     # objects are spatially close
    SUPPORTS = 3    # surface supports object
    CONTAINS = 4    # room/container contains object
    CONNECTED = 5   # places are traversable
    ABOVE = 6       # object is above another
    BELOW = 7       # object is below another
    LEFT_OF = 8     # relative spatial
    RIGHT_OF = 9    # relative spatial
    PART_OF = 10    # hierarchical
    UNKNOWN = 11


# String → EdgeType mapping for parsing
EDGE_TYPE_MAP = {
    "isOn": EdgeType.IS_ON,
    "isIn": EdgeType.IS_IN,
    "nearBy": EdgeType.NEAR_BY,
    "supports": EdgeType.SUPPORTS,
    "contains": EdgeType.CONTAINS,
    "connected": EdgeType.CONNECTED,
    "above": EdgeType.ABOVE,
    "below": EdgeType.BELOW,
    "leftOf": EdgeType.LEFT_OF,
    "rightOf": EdgeType.RIGHT_OF,
    "partOf": EdgeType.PART_OF,
}


@dataclass
class SceneNode:
    """A single node in the scene graph."""
    node_id: int
    semantic_label: int  # integer class ID
    label_name: str  # human-readable name (e.g., "mug", "table")
    position: np.ndarray  # [x, y, z] in world frame
    bounding_box: np.ndarray  # [x, y, z, w, h, d] center + dimensions
    layer: NodeLayer
    # Optional: visual crop for DINOv2 encoding (set during perception)
    visual_crop: Optional[np.ndarray] = None  # [H, W, 3] RGB crop
    # Optional: precomputed visual feature
    visual_feature: Optional[np.ndarray] = None  # [1024] DINOv2 feature
    # Additional properties
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class SceneEdge:
    """A single edge in the scene graph."""
    source_id: int
    target_id: int
    edge_type: EdgeType
    weight: float = 1.0


@dataclass
class SceneGraph:
    """A complete 3D scene graph snapshot at time t.

    This is the intermediate representation between Hydra output
    and the PyG Data object used by the world model.
    """
    nodes: List[SceneNode]
    edges: List[SceneEdge]
    timestamp: float = 0.0

    def __post_init__(self):
        self._id_to_idx: Dict[int, int] = {}
        for idx, node in enumerate(self.nodes):
            self._id_to_idx[node.node_id] = idx

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_node_by_id(self, node_id: int) -> Optional[SceneNode]:
        idx = self._id_to_idx.get(node_id)
        return self.nodes[idx] if idx is not None else None

    def get_node_idx(self, node_id: int) -> Optional[int]:
        return self._id_to_idx.get(node_id)

    def get_label_names(self) -> List[str]:
        return [n.label_name for n in self.nodes]

    def find_nodes_by_label(self, label_name: str) -> List[SceneNode]:
        return [n for n in self.nodes if n.label_name.lower() == label_name.lower()]

    def to_pyg(self, node_features: Optional[torch.Tensor] = None) -> Data:
        """Convert to PyTorch Geometric Data object.

        Args:
            node_features: Precomputed node features [num_nodes, feat_dim].
                If None, uses raw position + label as minimal features.

        Returns:
            PyG Data object ready for the world model.
        """
        # Node features
        if node_features is not None:
            x = node_features
        else:
            # Minimal features: position + one-hot layer + label
            feats = []
            for node in self.nodes:
                pos = torch.tensor(node.position, dtype=torch.float32)
                layer_oh = torch.zeros(4)
                layer_oh[node.layer] = 1.0
                label = torch.tensor([node.semantic_label], dtype=torch.float32)
                feats.append(torch.cat([pos, layer_oh, label]))
            x = torch.stack(feats)

        # Edge index (bidirectional)
        src_list, tgt_list = [], []
        edge_types = []
        for edge in self.edges:
            src_idx = self._id_to_idx.get(edge.source_id)
            tgt_idx = self._id_to_idx.get(edge.target_id)
            if src_idx is not None and tgt_idx is not None:
                # Forward edge
                src_list.append(src_idx)
                tgt_list.append(tgt_idx)
                edge_types.append(edge.edge_type.value)
                # Reverse edge
                src_list.append(tgt_idx)
                tgt_list.append(src_idx)
                edge_types.append(edge.edge_type.value)

        if len(src_list) > 0:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros(0, dtype=torch.long)

        # Store metadata
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.nodes),
        )

        # Store node positions separately for spatial reasoning
        positions = torch.tensor(
            [n.position for n in self.nodes], dtype=torch.float32
        )
        data.pos = positions

        # Store semantic labels for evaluation
        data.semantic_labels = torch.tensor(
            [n.semantic_label for n in self.nodes], dtype=torch.long
        )

        # Store label names as metadata
        data.label_names = [n.label_name for n in self.nodes]

        return data

    def to_adjacency_matrix(self) -> torch.Tensor:
        """Return dense adjacency matrix [N, N] with edge type values."""
        n = self.num_nodes
        adj = torch.zeros(n, n, dtype=torch.long)
        for edge in self.edges:
            src_idx = self._id_to_idx.get(edge.source_id)
            tgt_idx = self._id_to_idx.get(edge.target_id)
            if src_idx is not None and tgt_idx is not None:
                adj[src_idx, tgt_idx] = edge.edge_type.value + 1  # +1 so 0 = no edge
                adj[tgt_idx, src_idx] = edge.edge_type.value + 1
        return adj

    @classmethod
    def from_dict(cls, data: dict) -> "SceneGraph":
        """Construct from a serialized dictionary (e.g., JSON)."""
        nodes = []
        for n in data["nodes"]:
            nodes.append(SceneNode(
                node_id=n["id"],
                semantic_label=n["semantic_label"],
                label_name=n["label_name"],
                position=np.array(n["position"]),
                bounding_box=np.array(n["bounding_box"]),
                layer=NodeLayer(n["layer"]),
                properties=n.get("properties", {}),
            ))

        edges = []
        for e in data["edges"]:
            edge_type = EDGE_TYPE_MAP.get(e.get("type", ""), EdgeType.UNKNOWN)
            edges.append(SceneEdge(
                source_id=e["source"],
                target_id=e["target"],
                edge_type=edge_type,
                weight=e.get("weight", 1.0),
            ))

        return cls(
            nodes=nodes,
            edges=edges,
            timestamp=data.get("timestamp", 0.0),
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON export."""
        inv_edge_map = {v: k for k, v in EDGE_TYPE_MAP.items()}
        return {
            "timestamp": self.timestamp,
            "nodes": [
                {
                    "id": n.node_id,
                    "semantic_label": n.semantic_label,
                    "label_name": n.label_name,
                    "position": n.position.tolist(),
                    "bounding_box": n.bounding_box.tolist(),
                    "layer": n.layer.value,
                    "properties": n.properties,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": inv_edge_map.get(e.edge_type, "unknown"),
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }
