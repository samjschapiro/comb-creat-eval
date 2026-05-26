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

# Athene-RM-8B ships a custom architecture (CustomAutoModelForSequenceClassification)
# but no remote modeling code, so transformers would silently fall back to a
# generic 2-label LlamaForSequenceClassification head (wrong reward). We
# reproduce Nexusflow's class verbatim: LlamaModel backbone + a scalar v_head,
# reward read at the last CLS token (id 128003) appended after the chat
# template. Built lazily so non-Athene scorers don't import LlamaModel.
_ATHENE_CLASS = None


def _athene_class():
    global _ATHENE_CLASS
    if _ATHENE_CLASS is not None:
        return _ATHENE_CLASS
    from torch import nn
    from transformers import LlamaModel, LlamaPreTrainedModel

    class AtheneForSequenceClassification(LlamaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.model = LlamaModel(config)
            self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
            self.CLS_ID = 128003
            self.post_init()

        def forward(self, input_ids=None, attention_mask=None, position_ids=None):
            # last_hidden_state == hidden_states[-1] numerically, but does
            # not retain all 33 layers' activations (large for the 64
            # long-sequence RM pass). Faithful, just memory-frugal.
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            rewards = self.v_head(out.last_hidden_state).squeeze(-1)
            scores = []
            for i in range(int(input_ids.shape[0])):
                c_inds = (input_ids[i] == self.CLS_ID).nonzero()
                # CLS is appended explicitly in score(), so it is always
                # present; guard anyway so one pathological row can't kill
                # a long run (fall back to the last token).
                c_ind = c_inds[-1].item() if c_inds.numel() else -1
                scores.append(rewards[i, c_ind])
            return {"scores": torch.stack(scores)}

    _ATHENE_CLASS = AtheneForSequenceClassification
    return _ATHENE_CLASS


@dataclass
class AppropriatenessScorer:
    model_name: str
    device: str = "cuda"
    load_in_4bit: bool = False
    max_length: int = 1024
    _tokenizer: object = field(default=None, init=False, repr=False)
    _model: object = field(default=None, init=False, repr=False)
    _use_chat_template: bool = field(default=False, init=False, repr=False)
    _is_athene: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        arch = getattr(
            AutoConfig.from_pretrained(self.model_name), "architectures", None
        ) or []
        self._is_athene = "CustomAutoModelForSequenceClassification" in arch

        load_kwargs: dict = {"torch_dtype": torch.bfloat16}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["device_map"] = self.device

        if self._is_athene:
            # Custom scalar-reward head; standard Auto* would load a wrong
            # generic 2-label head. Read at the CLS token in score().
            self._model = _athene_class().from_pretrained(
                self.model_name, **load_kwargs
            ).eval()
            if not self.load_in_4bit:
                self._model = self._model.to(self.device)
            self._use_chat_template = True
            return

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **load_kwargs
        ).eval()
        if not self.load_in_4bit:
            self._model = self._model.to(self.device)
        # Decoder-based sequence-classification RMs (Llama/Mistral) need
        # config.pad_token_id to locate the last non-pad token per row;
        # without it, batched (>1) scoring raises "Cannot handle batch
        # sizes > 1 if no padding token is defined."
        if self._model.config.pad_token_id is None:
            self._model.config.pad_token_id = self._tokenizer.pad_token_id

        self._use_chat_template = getattr(self._tokenizer, "chat_template", None) is not None

    @torch.no_grad()
    def score(self, prompts: list[str], completions: list[str]) -> np.ndarray:
        if len(prompts) != len(completions):
            raise ValueError(
                f"FATAL: shape mismatch: {len(prompts)} prompts vs "
                f"{len(completions)} completions"
            )

        if self._is_athene:
            # Nexusflow preprocess: chat-template the [user, assistant]
            # turn, then the model reads its reward at the CLS token.
            # Appending CLS as a *string* before truncation drops it for
            # long sequences (CLS is right-truncated -> empty c_inds).
            # Instead tokenize without CLS, right-truncate to max_len-1,
            # then append the CLS token id so it is ALWAYS the last token
            # regardless of length. max_len 4096 matches their pipeline.
            cls_id = self._tokenizer.cls_token_id
            max_len = 4096
            seqs = []
            for p, c in zip(prompts, completions):
                text = self._tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": p},
                        {"role": "assistant", "content": c},
                    ],
                    tokenize=False,
                )
                ids = self._tokenizer(
                    text, add_special_tokens=False, truncation=True,
                    max_length=max_len - 1,
                )["input_ids"]
                seqs.append(ids + [cls_id])
            enc = self._tokenizer.pad(
                {"input_ids": seqs}, padding=True, return_tensors="pt"
            )
            dev = self._model.device
            out = self._model(
                input_ids=enc["input_ids"].to(dev),
                attention_mask=enc["attention_mask"].to(dev),
            )
            scores = out["scores"].float().cpu().numpy()
            if scores.ndim == 0:
                scores = scores.reshape(1)
            return scores

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
