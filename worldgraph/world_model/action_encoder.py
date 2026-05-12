"""CLIP-based action encoder. Converts text action descriptions into
fixed-size embeddings for conditioning the transition model.
"""

from typing import List, Union

import torch
import torch.nn as nn


class ActionEncoder(nn.Module):
    """Encodes high-level text actions using a frozen CLIP text encoder.

    Actions like "find keys", "navigate to kitchen", "pick up mug" are
    converted to dense embeddings that condition the transition model.

    Uses CLIP ViT-L/14 text encoder (768-d output) for strong
    vision-language alignment.
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        super().__init__()
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        """Lazy load to avoid import-time GPU allocation."""
        if self._model is not None:
            return

        from transformers import CLIPTextModel, CLIPTokenizer

        self._tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self._model = CLIPTextModel.from_pretrained(self.model_name)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

    @property
    def output_dim(self) -> int:
        return 768  # CLIP ViT-L/14 text encoder output

    def to(self, device):
        super().to(device)
        self._lazy_load()
        self._model = self._model.to(device)
        return self

    @torch.no_grad()
    def forward(self, actions: Union[str, List[str]]) -> torch.Tensor:
        """Encode action text(s) into embeddings.

        Args:
            actions: Single action string or list of action strings.

        Returns:
            [B, 768] tensor of action embeddings.
        """
        self._lazy_load()
        device = next(self._model.parameters()).device

        if isinstance(actions, str):
            actions = [actions]

        inputs = self._tokenizer(
            actions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self._model(**inputs)
        return outputs.pooler_output  # [B, 768]

    @torch.no_grad()
    def encode_action_vocabulary(self, actions: List[str]) -> torch.Tensor:
        """Pre-encode a fixed action vocabulary for fast lookup.

        Args:
            actions: List of all possible action strings.

        Returns:
            [len(actions), 768] tensor of precomputed embeddings.
        """
        return self.forward(actions)
