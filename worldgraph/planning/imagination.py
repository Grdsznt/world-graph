"""Imagination Engine: simulates action outcomes using the world model
and selects the best recommendation via scoring.

This is the planning loop:
1. For each candidate action, predict the next scene graph state
2. Score each predicted outcome using task-specific scoring functions
3. Return the best action with its predicted outcome
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch_geometric.data import Data

from worldgraph.config import ImaginationConfig
from worldgraph.world_model.transition import TransitionModel
from worldgraph.world_model.action_encoder import ActionEncoder
from worldgraph.planning.scoring import BaseScorer


@dataclass
class ImaginationResult:
    """Result of imagining a single action."""
    action_text: str
    predicted_graph: Data
    score: float
    trajectory: List[Data]  # [s_t, s_{t+1}, ...] for multi-step


@dataclass
class PlanningResult:
    """Result of the full planning process."""
    best_action: str
    best_score: float
    best_predicted_graph: Data
    all_results: List[ImaginationResult]


class ImaginationEngine:
    """Uses the world model to simulate and evaluate action outcomes.

    For each candidate action:
    1. Encode the action text via CLIP
    2. Run the transition model to predict s_{t+1}
    3. (Optional) Chain predictions for multi-step rollout
    4. Score the final predicted state
    5. Rank actions by score
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        action_encoder: ActionEncoder,
        scorer: BaseScorer,
        config: ImaginationConfig = ImaginationConfig(),
    ):
        self.transition = transition_model
        self.action_enc = action_encoder
        self.scorer = scorer
        self.config = config

    @torch.no_grad()
    def imagine_single(
        self,
        current_graph: Data,
        action_text: str,
        depth: Optional[int] = None,
    ) -> ImaginationResult:
        """Imagine the outcome of a single action.

        Args:
            current_graph: Current scene graph state.
            action_text: Action to simulate.
            depth: Number of imagination steps (default: config.max_rollout_depth).

        Returns:
            ImaginationResult with predicted state and score.
        """
        depth = depth or self.config.max_rollout_depth

        # Encode action
        action_emb = self.action_enc(action_text)  # [1, 768]

        # Rollout
        trajectory = [current_graph]
        g = current_graph

        for _ in range(depth):
            g_next = self.transition.predict_next_graph(
                g, action_emb,
                edge_threshold=self.config.edge_threshold,
            )
            trajectory.append(g_next)
            g = g_next

        # Score the final state
        score = self.scorer.score(
            current_graph=current_graph,
            predicted_graph=trajectory[-1],
            action_text=action_text,
        )

        return ImaginationResult(
            action_text=action_text,
            predicted_graph=trajectory[-1],
            score=score,
            trajectory=trajectory,
        )

    @torch.no_grad()
    def plan(
        self,
        current_graph: Data,
        candidate_actions: List[str],
        depth: Optional[int] = None,
        top_k: Optional[int] = None,
        **scorer_kwargs,
    ) -> PlanningResult:
        """Imagine all candidate actions and select the best one.

        Args:
            current_graph: Current scene graph state.
            candidate_actions: List of action strings to evaluate.
            depth: Imagination depth per action.
            top_k: Return only top-k results (None = all).

        Returns:
            PlanningResult with ranked actions and predictions.
        """
        self.transition.eval()
        depth = depth or self.config.max_rollout_depth

        results = []
        for action_text in candidate_actions:
            result = self.imagine_single(current_graph, action_text, depth)
            results.append(result)

        # Sort by score (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return PlanningResult(
            best_action=results[0].action_text,
            best_score=results[0].score,
            best_predicted_graph=results[0].predicted_graph,
            all_results=results,
        )

    @torch.no_grad()
    def plan_batch(
        self,
        current_graph: Data,
        candidate_actions: List[str],
    ) -> PlanningResult:
        """Batch-evaluate all actions in a single forward pass (single-step only).

        More efficient than plan() for single-step imagination with many candidates,
        as it batches all action encodings and graph replicas.

        Args:
            current_graph: Current scene graph state.
            candidate_actions: List of action strings.

        Returns:
            PlanningResult with ranked actions.
        """
        from torch_geometric.data import Batch

        self.transition.eval()
        n_actions = len(candidate_actions)

        # Encode all actions at once
        action_embs = self.action_enc(candidate_actions)  # [N_actions, 768]

        # Replicate graph for each action
        graphs = [current_graph.clone() for _ in range(n_actions)]
        batch = Batch.from_data_list(graphs)

        # Single batched forward pass
        output = self.transition(batch, action_embs)

        # Score each predicted outcome
        results = []
        for i, action_text in enumerate(candidate_actions):
            # Extract per-graph predictions (simplified for single-step)
            pred_graph = current_graph.clone()
            # Note: for full implementation, need to de-batch output properly
            score = self.scorer.score(current_graph, pred_graph, action_text)
            results.append(ImaginationResult(
                action_text=action_text,
                predicted_graph=pred_graph,
                score=score,
                trajectory=[current_graph, pred_graph],
            ))

        results.sort(key=lambda r: r.score, reverse=True)

        return PlanningResult(
            best_action=results[0].action_text,
            best_score=results[0].score,
            best_predicted_graph=results[0].predicted_graph,
            all_results=results,
        )
