"""Hydra JSON → PyTorch Geometric preprocessing.

Parses a Hydra Scene Graph exported as JSON, extracts Layer 2 (Places)
and Layer 5 (Objects), and converts to a PyG Data object with:
- x: [N, 515] = concat(SigLIP 512-d features, 3D position)
- edge_index: [2, E] bidirectional edges
- edge_attr: [E] edge type indices
- pos: [N, 3] 3D coordinates
- node_type: [N] 0=object, 1=place
- node_id_map: dict mapping original Hydra node IDs → GNN indices

JSON Schema (expected input):
{
  "layers": {
    "5": {  // Objects
      "nodes": [
        {
          "id": "O42",
          "semantic_label": 5,
          "name": "mug",
          "position": [1.2, 0.8, 0.5],
          "bounding_box": [1.2, 0.8, 0.5, 0.1, 0.1, 0.15],
          "features": [0.12, -0.34, ...]  // 512-d SigLIP embedding
        }
      ]
    },
    "2": {  // Places
      "nodes": [
        {
          "id": "P7",
          "position": [2.0, 1.5, 0.0],
          "features": [0.05, 0.22, ...]  // 512-d SigLIP embedding
        }
      ]
    }
  },
  "edges": [
    {"source": "O42", "target": "P7", "type": "is_in"},
    {"source": "O42", "target": "O43", "type": "near_by"}
  ]
}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data

# Hydra layer IDs
HYDRA_LAYER_OBJECTS = "5"
HYDRA_LAYER_PLACES = "2"
HYDRA_LAYER_ROOMS = "4"
HYDRA_LAYER_MESH_PLACES = "1"

# Edge type string → integer mapping
EDGE_TYPE_MAP = {
    "is_on": 1,
    "is_in": 2,
    "near_by": 3,
    "supports": 4,
    "contains": 5,
    "connected": 6,
    "above": 7,
    "below": 8,
    "left_of": 9,
    "right_of": 10,
    "part_of": 11,
}


def parse_hydra_json(
    json_path: Union[str, Path, dict],
    feature_dim: int = 512,
    include_places: bool = True,
    include_rooms: bool = False,
) -> Tuple[Data, Dict[str, int]]:
    """Parse a Hydra Scene Graph JSON into a PyG Data object.

    Args:
        json_path: Path to JSON file, or pre-loaded dict.
        feature_dim: Dimensionality of node feature vectors (SigLIP = 512).
        include_places: Whether to include Layer 2 (Places) nodes.
        include_rooms: Whether to include Layer 4 (Rooms) nodes.

    Returns:
        Tuple of:
        - data: PyG Data object with x, edge_index, edge_attr, pos, node_type
        - node_id_map: Dict mapping original Hydra node ID strings → GNN indices
    """
    # Load JSON
    if isinstance(json_path, dict):
        raw = json_path
    else:
        with open(json_path) as f:
            raw = json.load(f)

    layers = raw.get("layers", {})
    raw_edges = raw.get("edges", [])

    # ── Parse nodes ──
    node_features = []
    node_positions = []
    node_types = []
    node_names = []
    node_semantic_labels = []
    node_id_map: Dict[str, int] = {}
    gnn_idx = 0

    # Layer 5: Objects (node_type = 0)
    if HYDRA_LAYER_OBJECTS in layers:
        for node in layers[HYDRA_LAYER_OBJECTS].get("nodes", []):
            node_id = str(node["id"])
            pos = np.array(node.get("position", [0, 0, 0]), dtype=np.float32)[:3]

            # SigLIP features
            feat = np.array(node.get("features", np.zeros(feature_dim)), dtype=np.float32)
            if len(feat) < feature_dim:
                feat = np.pad(feat, (0, feature_dim - len(feat)))
            elif len(feat) > feature_dim:
                feat = feat[:feature_dim]

            node_features.append(feat)
            node_positions.append(pos)
            node_types.append(0)  # object
            node_names.append(node.get("name", f"object_{gnn_idx}"))
            node_semantic_labels.append(node.get("semantic_label", 0))
            node_id_map[node_id] = gnn_idx
            gnn_idx += 1

    # Layer 2: Places (node_type = 1)
    if include_places and HYDRA_LAYER_PLACES in layers:
        for node in layers[HYDRA_LAYER_PLACES].get("nodes", []):
            node_id = str(node["id"])
            pos = np.array(node.get("position", [0, 0, 0]), dtype=np.float32)[:3]

            feat = np.array(node.get("features", np.zeros(feature_dim)), dtype=np.float32)
            if len(feat) < feature_dim:
                feat = np.pad(feat, (0, feature_dim - len(feat)))
            elif len(feat) > feature_dim:
                feat = feat[:feature_dim]

            node_features.append(feat)
            node_positions.append(pos)
            node_types.append(1)  # place
            node_names.append(node.get("name", f"place_{gnn_idx}"))
            node_semantic_labels.append(node.get("semantic_label", -1))
            node_id_map[node_id] = gnn_idx
            gnn_idx += 1

    # Layer 4: Rooms (node_type = 2) — optional
    if include_rooms and HYDRA_LAYER_ROOMS in layers:
        for node in layers[HYDRA_LAYER_ROOMS].get("nodes", []):
            node_id = str(node["id"])
            pos = np.array(node.get("position", [0, 0, 0]), dtype=np.float32)[:3]
            feat = np.array(node.get("features", np.zeros(feature_dim)), dtype=np.float32)
            if len(feat) < feature_dim:
                feat = np.pad(feat, (0, feature_dim - len(feat)))

            node_features.append(feat)
            node_positions.append(pos)
            node_types.append(2)
            node_names.append(node.get("name", f"room_{gnn_idx}"))
            node_semantic_labels.append(node.get("semantic_label", -1))
            node_id_map[node_id] = gnn_idx
            gnn_idx += 1

    if len(node_features) == 0:
        # Return empty graph
        return Data(
            x=torch.zeros(0, feature_dim + 3),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, dtype=torch.long),
            pos=torch.zeros(0, 3),
        ), {}

    # ── Build node feature matrix: x = [SigLIP_512 | pos_3] ──
    features_np = np.stack(node_features)         # [N, 512]
    positions_np = np.stack(node_positions)        # [N, 3]
    x_np = np.concatenate([features_np, positions_np], axis=1)  # [N, 515]

    x = torch.tensor(x_np, dtype=torch.float32)
    pos = torch.tensor(positions_np, dtype=torch.float32)
    node_type = torch.tensor(node_types, dtype=torch.long)
    semantic_labels = torch.tensor(node_semantic_labels, dtype=torch.long)

    # ── Parse edges (bidirectional) ──
    src_list, tgt_list, edge_types = [], [], []
    for edge in raw_edges:
        src_id = str(edge["source"])
        tgt_id = str(edge["target"])
        if src_id not in node_id_map or tgt_id not in node_id_map:
            continue  # skip edges to nodes we didn't include

        src_idx = node_id_map[src_id]
        tgt_idx = node_id_map[tgt_id]
        edge_type_str = edge.get("type", "unknown").lower().replace("-", "_").replace(" ", "_")
        edge_type_int = EDGE_TYPE_MAP.get(edge_type_str, 0)

        # Forward
        src_list.append(src_idx)
        tgt_list.append(tgt_idx)
        edge_types.append(edge_type_int)
        # Reverse
        src_list.append(tgt_idx)
        tgt_list.append(src_idx)
        edge_types.append(edge_type_int)

    if len(src_list) > 0:
        edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, dtype=torch.long)

    # ── Construct Data object ──
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        node_type=node_type,
        semantic_labels=semantic_labels,
    )
    data.label_names = node_names
    data.num_nodes = x.size(0)

    return data, node_id_map


def create_sample_hydra_json(
    num_objects: int = 8,
    num_places: int = 3,
    feature_dim: int = 512,
    seed: int = 42,
) -> dict:
    """Generate a synthetic Hydra JSON for testing.

    Creates a realistic kitchen-like scene graph with objects on surfaces,
    objects in rooms, and place connectivity.

    Args:
        num_objects: Number of object nodes.
        num_places: Number of place nodes.
        feature_dim: Feature vector dimensionality.
        seed: Random seed.

    Returns:
        Dict in Hydra JSON format.
    """
    rng = np.random.RandomState(seed)

    object_names = [
        "mug", "plate", "fork", "knife", "bottle",
        "bowl", "cup", "spoon", "pan", "cutting_board",
        "toaster", "kettle", "apple", "banana", "bread",
    ]
    place_names = ["kitchen_center", "living_room", "hallway", "bathroom", "bedroom"]

    # Generate objects
    objects = []
    for i in range(num_objects):
        name = object_names[i % len(object_names)]
        objects.append({
            "id": f"O{i}",
            "semantic_label": i % 20,
            "name": name,
            "position": rng.randn(3).tolist(),
            "bounding_box": rng.randn(6).tolist(),
            "features": rng.randn(feature_dim).tolist(),
        })

    # Generate places
    places = []
    for i in range(num_places):
        name = place_names[i % len(place_names)]
        places.append({
            "id": f"P{i}",
            "semantic_label": 100 + i,
            "name": name,
            "position": (rng.randn(3) * 3).tolist(),
            "features": rng.randn(feature_dim).tolist(),
        })

    # Generate edges
    edges = []
    # Objects → places (containment)
    for i in range(num_objects):
        place_idx = i % num_places
        edges.append({
            "source": f"O{i}",
            "target": f"P{place_idx}",
            "type": "is_in",
        })

    # Object→object (proximity)
    for i in range(num_objects - 1):
        if rng.rand() > 0.5:
            edges.append({
                "source": f"O{i}",
                "target": f"O{i + 1}",
                "type": "near_by",
            })

    # Place connectivity
    for i in range(num_places - 1):
        edges.append({
            "source": f"P{i}",
            "target": f"P{i + 1}",
            "type": "connected",
        })

    return {
        "layers": {
            "5": {"nodes": objects},
            "2": {"nodes": places},
        },
        "edges": edges,
    }


def export_pyg_to_hydra_json(data: Data, node_id_map: Dict[str, int], output_path: str):
    """Export a PyG Data object back to Hydra JSON format."""
    inv_map = {v: k for k, v in node_id_map.items()}
    inv_edge_map = {v: k for k, v in EDGE_TYPE_MAP.items()}

    objects, places = [], []
    for idx in range(data.num_nodes):
        node_id = inv_map.get(idx, f"N{idx}")
        node_type = data.node_type[idx].item() if hasattr(data, "node_type") else 0
        entry = {
            "id": node_id,
            "position": data.pos[idx].tolist() if hasattr(data, "pos") else [0, 0, 0],
            "features": data.x[idx, :512].tolist(),
        }
        if hasattr(data, "label_names"):
            entry["name"] = data.label_names[idx]
        if hasattr(data, "semantic_labels"):
            entry["semantic_label"] = data.semantic_labels[idx].item()

        if node_type == 0:
            objects.append(entry)
        else:
            places.append(entry)

    edges = []
    for i in range(data.edge_index.size(1)):
        src = data.edge_index[0, i].item()
        tgt = data.edge_index[1, i].item()
        etype = data.edge_attr[i].item() if data.edge_attr is not None else 0
        # Skip reverse edges (only keep one direction)
        if src < tgt:
            edges.append({
                "source": inv_map.get(src, f"N{src}"),
                "target": inv_map.get(tgt, f"N{tgt}"),
                "type": inv_edge_map.get(etype, "unknown"),
            })

    result = {
        "layers": {
            "5": {"nodes": objects},
            "2": {"nodes": places},
        },
        "edges": edges,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
