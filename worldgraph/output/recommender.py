"""LLM-powered recommendation generator.

Takes the best action and predicted scene graph from the imagination engine
and generates a polished natural language recommendation for the AR UI.

Uses LLaMA 3.1 8B Instruct at 4-bit quantization (~5GB VRAM).
"""

from typing import Dict, List, Optional

import torch

from worldgraph.config import OutputConfig
from worldgraph.planning.imagination import PlanningResult


SYSTEM_PROMPT = """You are an AR assistant that helps users with spatial tasks. 
Given a scene analysis and a recommended action, generate a brief, friendly, 
actionable UI recommendation. Be specific about locations and objects. 
Keep it to 1-2 sentences maximum."""


class RecommendationGenerator:
    """Generates natural language recommendations from planning results.

    Uses a quantized LLM to convert structured planning outputs into
    human-friendly AR UI text.
    """

    def __init__(self, config: OutputConfig = OutputConfig()):
        self.config = config
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        """Lazy load the LLM to avoid import-time GPU allocation."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        if self.config.llm_quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.llm_quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()

    @torch.no_grad()
    def generate(
        self,
        planning_result: PlanningResult,
        task_context: str = "",
        scene_description: Optional[str] = None,
    ) -> str:
        """Generate a natural language recommendation.

        Args:
            planning_result: Output from the imagination engine.
            task_context: Additional context (e.g., "user is looking for keys").
            scene_description: Optional text description of current scene.

        Returns:
            Natural language recommendation string.
        """
        self._lazy_load()

        # Build the prompt from planning results
        best = planning_result.all_results[0]

        # Extract scene info from predicted graph
        if scene_description is None:
            pred_names = getattr(best.predicted_graph, "label_names", [])
            scene_description = f"Objects in scene: {', '.join(pred_names[:20])}"

        # Format top results
        results_text = ""
        for i, result in enumerate(planning_result.all_results[:3]):
            results_text += f"  {i + 1}. {result.action_text} (confidence: {result.score:.0%})\n"

        user_prompt = f"""Scene: {scene_description}
Task: {task_context}
Best action: {best.action_text} (confidence: {best.score:.0%})
Top alternatives:
{results_text}
Generate a brief AR UI recommendation for the user."""

        # Build chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Tokenize
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        # Generate
        outputs = self._model.generate(
            input_ids,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        # Decode only the generated part
        generated = outputs[0][input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_simple(self, action: str, score: float, context: str = "") -> str:
        """Generate a recommendation from minimal inputs (no planning result needed).

        Useful for testing or when you just have an action string.
        """
        self._lazy_load()

        prompt = f"Action: {action}. Confidence: {score:.0%}. Context: {context}. Write a 1-sentence AR UI tip."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=64,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )

        generated = outputs[0][input_ids.shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class TemplateRecommender:
    """Lightweight alternative to LLM-based generation.

    Uses templates for fast, deterministic output without loading an LLM.
    Useful for testing or low-latency scenarios.
    """

    TEMPLATES = {
        "find": "Check the {location} — your {object} should be {relation} the {surface}.",
        "recipe": "You have the ingredients for {recipe}! Start by {first_step}.",
        "navigate": "Head to the {location}. It's about {distance} away.",
        "default": "Recommended action: {action} (confidence: {score:.0%})",
    }

    def generate(self, planning_result: PlanningResult, **kwargs) -> str:
        best = planning_result.all_results[0]
        action = best.action_text.lower()

        # Select template
        if any(kw in action for kw in ["find", "locate", "search", "where"]):
            template_key = "find"
        elif any(kw in action for kw in ["recipe", "cook", "make"]):
            template_key = "recipe"
        elif any(kw in action for kw in ["go to", "navigate", "move"]):
            template_key = "navigate"
        else:
            template_key = "default"

        return self.TEMPLATES[template_key].format(
            action=best.action_text,
            score=best.score,
            object="item",
            location="nearby area",
            relation="on",
            surface="surface",
            recipe=best.action_text,
            first_step="gathering ingredients",
            distance="a few steps",
        )
