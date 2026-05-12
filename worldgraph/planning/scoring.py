"""Task-specific scoring functions (reward proxies).

These replace a learned reward model with hand-designed heuristics
for specific AR/XR use cases. Each scorer evaluates how desirable
a predicted next-state is, given the current state and the action taken.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

import torch
from torch_geometric.data import Data


class BaseScorer(ABC):
    """Abstract base class for scoring predicted outcomes."""

    @abstractmethod
    def score(
        self,
        current_graph: Data,
        predicted_graph: Data,
        action_text: str,
        **kwargs,
    ) -> float:
        """Score a predicted outcome.

        Args:
            current_graph: Current scene graph state s_t.
            predicted_graph: Predicted next state s_{t+1}.
            action_text: The action that was "imagined".

        Returns:
            Score in [0, 1] where higher is better.
        """
        ...


class LostObjectScorer(BaseScorer):
    """Scorer for the lost object finding task.

    Evaluates whether executing an action would reveal the target object
    in a visible/accessible location.
    """

    def __init__(self, user_position: Optional[torch.Tensor] = None):
        self.user_position = user_position

    def score(
        self,
        current_graph: Data,
        predicted_graph: Data,
        action_text: str,
        target_object: str = "",
        **kwargs,
    ) -> float:
        """Score based on target object visibility and proximity.

        Scoring criteria:
        1. Does the target object exist in the predicted state? (+0.3)
        2. Is it connected via a "visible" edge type? (+0.3)
        3. How close is it to the user? (+0.4, distance-weighted)
        """
        if not target_object:
            # Try to extract target from action text
            target_object = self._extract_target(action_text)

        if not target_object:
            return 0.0

        score = 0.0

        # Check if target exists in predicted graph
        pred_names = getattr(predicted_graph, "label_names", [])
        target_lower = target_object.lower()

        matching_indices = [
            i for i, name in enumerate(pred_names) if target_lower in name.lower()
        ]

        if not matching_indices:
            return 0.0  # Object not found at all

        score += 0.3  # Object exists

        # Check connectivity (is it on a visible surface?)
        if predicted_graph.edge_index.size(1) > 0:
            for idx in matching_indices:
                # Find edges connected to this node
                mask = (predicted_graph.edge_index[0] == idx) | (
                    predicted_graph.edge_index[1] == idx
                )
                if mask.any():
                    score += 0.3  # Connected to something (likely a surface)
                    break

        # Distance-based score
        if (
            self.user_position is not None
            and hasattr(predicted_graph, "pos")
            and predicted_graph.pos is not None
        ):
            for idx in matching_indices:
                obj_pos = predicted_graph.pos[idx]
                dist = torch.norm(self.user_position - obj_pos).item()
                proximity_score = max(0.0, 1.0 - dist / 10.0)  # normalize by 10m
                score += 0.4 * proximity_score
                break  # use closest match

        return min(score, 1.0)

    @staticmethod
    def _extract_target(action_text: str) -> str:
        """Simple heuristic to extract target object from action text."""
        keywords = ["find", "locate", "search for", "look for", "where is", "where are"]
        text_lower = action_text.lower()
        for kw in keywords:
            if kw in text_lower:
                # Return everything after the keyword
                remainder = text_lower.split(kw, 1)[1].strip()
                # Remove articles
                for article in ["the ", "my ", "a ", "an "]:
                    if remainder.startswith(article):
                        remainder = remainder[len(article):]
                return remainder.strip().rstrip("?.")
        return ""


class RecipeScorer(BaseScorer):
    """Scorer for the kitchen recipe suggestion task.

    Evaluates whether the predicted state has all required ingredients
    accessible for a given recipe.
    """

    # Simple recipe database (expandable)
    RECIPES: Dict[str, Set[str]] = {
        "pasta carbonara": {"pasta", "egg", "bacon", "parmesan", "pepper"},
        "grilled cheese": {"bread", "cheese", "butter"},
        "salad": {"lettuce", "tomato", "cucumber", "dressing"},
        "omelette": {"egg", "cheese", "pepper", "onion"},
        "smoothie": {"banana", "milk", "yogurt", "berry"},
        "toast": {"bread", "butter"},
        "cereal": {"cereal", "milk"},
        "sandwich": {"bread", "meat", "cheese", "lettuce"},
        "soup": {"broth", "vegetable", "onion", "garlic"},
        "stir fry": {"rice", "vegetable", "oil", "soy sauce"},
    }

    def score(
        self,
        current_graph: Data,
        predicted_graph: Data,
        action_text: str,
        **kwargs,
    ) -> float:
        """Score based on ingredient availability for the suggested recipe.

        Scoring: fraction of required ingredients visible in the scene.
        """
        recipe_name = self._extract_recipe(action_text)
        if not recipe_name:
            return 0.0

        required = self.RECIPES.get(recipe_name, set())
        if not required:
            return 0.0

        # Check which ingredients are present in the current/predicted graph
        available_names = set()
        for graph in [current_graph, predicted_graph]:
            names = getattr(graph, "label_names", [])
            for name in names:
                available_names.add(name.lower())

        # Fuzzy matching: check if any available item contains the ingredient name
        found = set()
        for ingredient in required:
            for available in available_names:
                if ingredient in available or available in ingredient:
                    found.add(ingredient)
                    break

        return len(found) / len(required)

    def suggest_best_recipe(
        self,
        graph: Data,
    ) -> List[tuple]:
        """Given a scene graph, rank all recipes by ingredient availability.

        Returns:
            List of (recipe_name, score, missing_ingredients) tuples, sorted by score.
        """
        available_names = set()
        names = getattr(graph, "label_names", [])
        for name in names:
            available_names.add(name.lower())

        results = []
        for recipe_name, required in self.RECIPES.items():
            found = set()
            for ingredient in required:
                for available in available_names:
                    if ingredient in available or available in ingredient:
                        found.add(ingredient)
                        break
            missing = required - found
            score = len(found) / len(required) if required else 0.0
            results.append((recipe_name, score, missing))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def _extract_recipe(action_text: str) -> str:
        """Extract recipe name from action text."""
        text_lower = action_text.lower()
        for recipe in RecipeScorer.RECIPES:
            if recipe in text_lower:
                return recipe
        return ""


class CompositeScorer(BaseScorer):
    """Combines multiple scorers with weighted average."""

    def __init__(self, scorers: Dict[str, tuple]):
        """
        Args:
            scorers: Dict mapping name → (scorer_instance, weight).
        """
        self.scorers = scorers

    def score(
        self,
        current_graph: Data,
        predicted_graph: Data,
        action_text: str,
        **kwargs,
    ) -> float:
        total_weight = sum(w for _, w in self.scorers.values())
        weighted_sum = 0.0

        for name, (scorer, weight) in self.scorers.items():
            s = scorer.score(current_graph, predicted_graph, action_text, **kwargs)
            weighted_sum += s * weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
