"""Hydra spark_dsg → WorldGraph SceneGraph converter.

Bridges Hydra's C++/Python scene graph output into the WorldGraph
data structures for world model processing.
"""

from typing import Optional

import numpy as np

from worldgraph.scene_graph.graph_types import (
    SceneGraph,
    SceneNode,
    SceneEdge,
    NodeLayer,
    EdgeType,
    EDGE_TYPE_MAP,
)


# Mapping from Hydra's DsgLayers to WorldGraph NodeLayer
# spark_dsg.DsgLayers: MESH_PLACES=1, PLACES=2, ROOMS=4, OBJECTS=5, BUILDINGS=6
HYDRA_LAYER_MAP = {
    1: NodeLayer.PLACE,     # MESH_PLACES
    2: NodeLayer.PLACE,     # PLACES
    4: NodeLayer.ROOM,      # ROOMS
    5: NodeLayer.OBJECT,    # OBJECTS
    6: NodeLayer.BUILDING,  # BUILDINGS
}


def hydra_to_scene_graph(
    dsg,
    include_places: bool = False,
    include_rooms: bool = True,
) -> SceneGraph:
    """Convert a Hydra spark_dsg DynamicSceneGraph to a WorldGraph SceneGraph.

    Args:
        dsg: A spark_dsg.DynamicSceneGraph object from Hydra.
        include_places: Whether to include place/mesh_places nodes.
        include_rooms: Whether to include room nodes.

    Returns:
        SceneGraph with all extracted nodes and edges.
    """
    try:
        import spark_dsg
    except ImportError:
        raise ImportError(
            "spark_dsg is required for Hydra integration. "
            "Install it with: pip install -e /path/to/Hydra"
        )

    nodes = []
    node_id_set = set()

    # Define which layers to include
    layers_to_include = [spark_dsg.DsgLayers.OBJECTS]
    if include_rooms:
        layers_to_include.append(spark_dsg.DsgLayers.ROOMS)
    if include_places:
        layers_to_include.extend([
            spark_dsg.DsgLayers.PLACES,
            spark_dsg.DsgLayers.MESH_PLACES,
        ])

    # Extract nodes from each layer
    for layer_id in layers_to_include:
        try:
            layer = dsg.get_layer(layer_id)
        except Exception:
            continue

        wg_layer = HYDRA_LAYER_MAP.get(layer_id, NodeLayer.OBJECT)

        for node_id_raw, node in layer.nodes.items():
            attrs = node.attributes

            # Extract position
            position = np.array(attrs.position, dtype=np.float32)
            if len(position) < 3:
                position = np.zeros(3, dtype=np.float32)

            # Extract bounding box (center + dimensions)
            try:
                bbox = np.array(attrs.bounding_box, dtype=np.float32)
                if len(bbox) < 6:
                    # Pad with position and default size
                    bbox = np.concatenate([position, np.ones(3, dtype=np.float32) * 0.1])
            except (AttributeError, TypeError):
                bbox = np.concatenate([position, np.ones(3, dtype=np.float32) * 0.1])

            # Extract semantic label
            try:
                semantic_label = int(attrs.semantic_label)
            except (AttributeError, TypeError):
                semantic_label = 0

            # Extract label name
            try:
                label_name = str(attrs.name) if hasattr(attrs, "name") else f"object_{semantic_label}"
            except (AttributeError, TypeError):
                label_name = f"object_{semantic_label}"

            # Convert node ID to int
            node_id = int(node_id_raw) if not isinstance(node_id_raw, int) else node_id_raw

            nodes.append(SceneNode(
                node_id=node_id,
                semantic_label=semantic_label,
                label_name=label_name,
                position=position,
                bounding_box=bbox[:6],
                layer=wg_layer,
            ))
            node_id_set.add(node_id)

    # Extract edges
    edges = []
    for edge in dsg.edges:
        src_id = int(edge.source) if not isinstance(edge.source, int) else edge.source
        tgt_id = int(edge.target) if not isinstance(edge.target, int) else edge.target

        # Only include edges between nodes we extracted
        if src_id not in node_id_set or tgt_id not in node_id_set:
            continue

        # Determine edge type
        edge_type = _infer_edge_type(dsg, edge, nodes, src_id, tgt_id)

        edges.append(SceneEdge(
            source_id=src_id,
            target_id=tgt_id,
            edge_type=edge_type,
        ))

    return SceneGraph(nodes=nodes, edges=edges)


def _infer_edge_type(dsg, edge, nodes, src_id, tgt_id) -> EdgeType:
    """Infer the semantic edge type from Hydra's edge and node context.

    Hydra doesn't explicitly label edge types, so we infer from
    the layer relationship:
    - Object → Room: IS_IN (containment)
    - Object → Object (same room): NEAR_BY
    - Place → Room: IS_IN
    - Place → Place: CONNECTED (traversable)
    """
    src_node = next((n for n in nodes if n.node_id == src_id), None)
    tgt_node = next((n for n in nodes if n.node_id == tgt_id), None)

    if src_node is None or tgt_node is None:
        return EdgeType.UNKNOWN

    # Inter-layer edges (hierarchical)
    if src_node.layer != tgt_node.layer:
        if src_node.layer == NodeLayer.OBJECT and tgt_node.layer == NodeLayer.ROOM:
            return EdgeType.IS_IN
        if src_node.layer == NodeLayer.ROOM and tgt_node.layer == NodeLayer.OBJECT:
            return EdgeType.CONTAINS
        if src_node.layer == NodeLayer.PLACE and tgt_node.layer == NodeLayer.ROOM:
            return EdgeType.IS_IN
        return EdgeType.PART_OF

    # Intra-layer edges
    if src_node.layer == NodeLayer.PLACE:
        return EdgeType.CONNECTED

    if src_node.layer == NodeLayer.OBJECT:
        # Infer spatial relation from positions
        delta = src_node.position - tgt_node.position
        if abs(delta[2]) > 0.3:  # significant vertical difference
            return EdgeType.ABOVE if delta[2] > 0 else EdgeType.BELOW
        dist = np.linalg.norm(delta)
        if dist < 0.5:
            return EdgeType.IS_ON  # very close, likely on same surface
        return EdgeType.NEAR_BY

    return EdgeType.UNKNOWN


def scene_graph_to_json(scene_graph: SceneGraph, output_path: str):
    """Export a SceneGraph to JSON for visualization/debugging."""
    import json
    with open(output_path, "w") as f:
        json.dump(scene_graph.to_dict(), f, indent=2)


def json_to_scene_graph(input_path: str) -> SceneGraph:
    """Load a SceneGraph from JSON."""
    import json
    with open(input_path) as f:
        data = json.load(f)
    return SceneGraph.from_dict(data)
