"""Appropriateness scoring (frozen RM) and SBERT embedding.

The RM is loaded once and frozen. We support two input formats:

- chat-template RMs (Llama / Mistral / InternLM-based): use
  tokenizer.apply_chat_template on a [user, assistant] conversation.
- pair RMs (DeBERTa-OpenAssistant style): tokenize (prompt, completion)
  as a sentence pair.

The path is selected by whether the tokenizer has a chat_template.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class AppropriatenessScorer:
    model_name: str
    device: str = "cuda"
    load_in_4bit: bool = False
    max_length: int = 1024
    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _use_chat_template: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict = {"torch_dtype": torch.bfloat16}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["device_map"] = self.device

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **load_kwargs
        ).eval()
        if not self.load_in_4bit:
            self._model = self._model.to(self.device)

        self._use_chat_template = getattr(self._tokenizer, "chat_template", None) is not None

    @torch.no_grad()
    def score(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        if len(prompts) != len(completions):
            raise ValueError(
                f"FATAL: shape mismatch: {len(prompts)} prompts vs "
                f"{len(completions)} completions"
            )

        if self._use_chat_template:
            texts = [
                self._tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": p},
                        {"role": "assistant", "content": c},
                    ],
                    tokenize=False,
                )
                for p, c in zip(prompts, completions)
            ]
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            inputs = self._tokenizer(
                prompts,
                completions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        logits = self._model(**inputs).logits
        scores = logits.squeeze(-1).float().cpu().numpy()
        if scores.ndim == 0:
            scores = scores.reshape(1)
        return scores


class SBERTEmbedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
