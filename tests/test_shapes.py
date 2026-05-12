"""Shape and dimension verification tests.

Run with: python -m pytest tests/test_shapes.py -v
Or simply: python tests/test_shapes.py
"""

import sys
import os

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldgraph.config import (
    WorldGraphConfig,
    NodeEncoderConfig,
    GraphEncoderConfig,
    TransitionModelConfig,
)
from worldgraph.scene_graph.graph_types import (
    SceneGraph,
    SceneNode,
    SceneEdge,
    NodeLayer,
    EdgeType,
)
from worldgraph.world_model.encoder import SceneGraphEncoder
from worldgraph.world_model.transition import TransitionModel


def make_dummy_scene_graph(num_nodes: int = 10, num_edges: int = 15) -> SceneGraph:
    """Create a synthetic scene graph for testing."""
    nodes = []
    for i in range(num_nodes):
        nodes.append(SceneNode(
            node_id=i,
            semantic_label=i % 20,
            label_name=f"object_{i}",
            position=np.random.randn(3).astype(np.float32),
            bounding_box=np.random.randn(6).astype(np.float32),
            layer=NodeLayer(i % 4),
        ))

    edges = []
    for _ in range(num_edges):
        src = np.random.randint(0, num_nodes)
        tgt = np.random.randint(0, num_nodes)
        if src != tgt:
            edges.append(SceneEdge(
                source_id=src,
                target_id=tgt,
                edge_type=EdgeType(np.random.randint(0, 12)),
            ))

    return SceneGraph(nodes=nodes, edges=edges)


def test_scene_graph_to_pyg():
    """Test SceneGraph → PyG Data conversion."""
    sg = make_dummy_scene_graph(10, 15)
    data = sg.to_pyg()

    assert data.x.shape[0] == 10, f"Expected 10 nodes, got {data.x.shape[0]}"
    assert data.x.dim() == 2, "Node features should be 2D"
    assert data.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert data.edge_attr.dim() == 1, "Edge attr should be 1D"
    assert data.pos.shape == (10, 3), f"Positions shape: {data.pos.shape}"
    print("✓ SceneGraph → PyG conversion: shapes correct")


def test_scene_graph_serialization():
    """Test JSON serialization round-trip."""
    sg = make_dummy_scene_graph(5, 8)
    d = sg.to_dict()
    sg2 = SceneGraph.from_dict(d)

    assert sg2.num_nodes == 5
    assert len(sg2.edges) == len(sg.edges)
    print("✓ SceneGraph JSON round-trip: correct")


def test_graph_encoder_shapes():
    """Test GPS encoder output shapes."""
    config = GraphEncoderConfig(hidden_dim=256, num_layers=2, num_heads=4, gat_heads=4)
    input_dim = 64  # small for testing
    encoder = SceneGraphEncoder(config=config, input_dim=input_dim)

    # Create fake input
    num_nodes = 20
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 40))
    edge_attr = torch.randint(0, 12, (40,))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    out = encoder(x, edge_index, edge_attr, batch)

    assert out.shape == (num_nodes, 256), f"Expected (20, 256), got {out.shape}"
    print(f"✓ GPS Encoder: input ({num_nodes}, {input_dim}) → output {out.shape}")


def test_transition_model_shapes():
    """Test full transition model forward pass shapes."""
    # Use small config for testing
    node_cfg = NodeEncoderConfig(
        visual_feat_dim=64,
        position_encoding_dim=24,
        bbox_feat_dim=6,
        semantic_emb_dim=12,
        layer_emb_dim=4,
    )
    graph_cfg = GraphEncoderConfig(hidden_dim=128, num_layers=2, num_heads=4, gat_heads=4)
    config = TransitionModelConfig(
        graph_encoder=graph_cfg,
        node_encoder=node_cfg,
        action_feat_dim=64,
        cross_attn_heads=4,
    )

    model = TransitionModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Transition model: {total_params:,} total params, {trainable_params:,} trainable")

    # Create fake input
    num_nodes = 15
    feat_dim = node_cfg.total_feat_dim  # 64+24+6+12+4 = 110
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, 30))
    edge_attr = torch.randint(0, 12, (30,))

    from torch_geometric.data import Data
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    action_emb = torch.randn(1, 64)

    output = model(graph, action_emb)

    assert output.predicted_node_features.shape == (num_nodes, feat_dim), \
        f"Node features: {output.predicted_node_features.shape}"
    assert output.node_existence_logits.shape == (num_nodes, 1), \
        f"Existence logits: {output.node_existence_logits.shape}"
    print(f"✓ Transition Model forward pass:")
    print(f"    Node features: {output.predicted_node_features.shape}")
    print(f"    Edge logits: {output.edge_logits.shape if output.edge_logits is not None else 'None'}")
    print(f"    Existence logits: {output.node_existence_logits.shape}")


def test_transition_predict_next_graph():
    """Test full predict_next_graph method."""
    node_cfg = NodeEncoderConfig(
        visual_feat_dim=64,
        position_encoding_dim=24,
        bbox_feat_dim=6,
        semantic_emb_dim=12,
        layer_emb_dim=4,
    )
    graph_cfg = GraphEncoderConfig(hidden_dim=128, num_layers=2, num_heads=4, gat_heads=4)
    config = TransitionModelConfig(
        graph_encoder=graph_cfg,
        node_encoder=node_cfg,
        action_feat_dim=64,
        cross_attn_heads=4,
    )

    model = TransitionModel(config)

    # Create graph with label names
    num_nodes = 10
    feat_dim = node_cfg.total_feat_dim
    x = torch.randn(num_nodes, feat_dim)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    edge_attr = torch.randint(0, 12, (20,))

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.randn(num_nodes, 3),
        semantic_labels=torch.arange(num_nodes),
        label_names=[f"object_{i}" for i in range(num_nodes)],
    )

    action_emb = torch.randn(1, 64)
    predicted = model.predict_next_graph(graph, action_emb, edge_threshold=0.3)

    print(f"✓ predict_next_graph:")
    print(f"    Input:  {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"    Output: {predicted.num_nodes} nodes, {predicted.edge_index.shape[1]} edges")


def test_full_config():
    """Test full config with real dimensions."""
    config = WorldGraphConfig()
    feat_dim = config.transition.node_encoder.total_feat_dim
    print(f"✓ Full config:")
    print(f"    Node feature dim: {feat_dim}")
    print(f"    Hidden dim: {config.transition.graph_encoder.hidden_dim}")
    print(f"    GPS layers: {config.transition.graph_encoder.num_layers}")
    print(f"    Action dim: {config.transition.action_feat_dim}")


if __name__ == "__main__":
    print("=" * 60)
    print("WorldGraph — Shape & Dimension Tests")
    print("=" * 60)

    test_scene_graph_to_pyg()
    print()

    test_scene_graph_serialization()
    print()

    test_graph_encoder_shapes()
    print()

    test_transition_model_shapes()
    print()

    test_transition_predict_next_graph()
    print()

    test_full_config()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
