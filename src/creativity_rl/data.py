"""Dataset loading and prompt clustering for MCNS-RL.

Loads open-ended prompts from HF datasets and assigns each to a
KMeans cluster over SBERT embeddings. Cluster IDs are used downstream
to route responses to the correct per-cluster archive.
"""

from __future__ import annotations

import numpy as np


def load_wildchat_prompts(
    n: int | None = None,
    seed: int = 17,
    hf_name: str = "allenai/WildChat-1M",
    split: str = "train",
) -> list[str]:
    """First user turn from English WildChat conversations, deduped.

    DARLING trains on a 10k WildChat prompt set. WildChat-1M has a
    `conversation` column (list of role/content turns) and a `language`
    field. We take the first user turn of English single-or-multi-turn
    conversations, dedupe, then subsample n.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_name, split=split, streaming=True)
    seen: set[str] = set()
    prompts: list[str] = []
    target = (n * 4) if n else 40000  # over-collect before dedupe/subsample
    for row in ds:
        if row.get("language") != "English":
            continue
        conv = row.get("conversation") or []
        first_user = next(
            (t.get("content") for t in conv if t.get("role") == "user"), None
        )
        if not first_user:
            continue
        key = first_user.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        prompts.append(first_user)
        if len(prompts) >= target:
            break
    if n is not None and n < len(prompts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(prompts), size=n, replace=False)
        prompts = [prompts[i] for i in idx]
    return prompts


def load_prompts(
    hf_name: str,
    split: str,
    text_field: str,
    n: int | None = None,
    seed: int = 17,
) -> list[str]:
    """Load and (optionally) subsample prompts from a HF dataset.

    text_field semantics:
      - regular column name: load list of strings from that column.
      - "messages": dataset is in conversational format; extract the
        first user-role message content from each row.
    """
    from datasets import load_dataset

    ds = load_dataset(hf_name, split=split)
    if text_field == "messages":
        if "messages" not in ds.column_names:
            raise ValueError(
                f"FATAL: text_field='messages' but no 'messages' column. "
                f"Available: {ds.column_names}"
            )
        prompts: list[str] = []
        for row in ds:
            msgs = row["messages"]
            user_msg = next(
                (m["content"] for m in msgs if m.get("role") == "user"),
                None,
            )
            if user_msg is None:
                raise ValueError(f"FATAL: no user message in row: {row}")
            prompts.append(user_msg)
    elif text_field in ds.column_names:
        prompts = list(ds[text_field])
    else:
        raise ValueError(
            f"FATAL: text_field '{text_field}' not in dataset columns "
            f"{ds.column_names}"
        )
    if n is not None and n < len(prompts):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(prompts), size=n, replace=False)
        prompts = [prompts[i] for i in idx]
    return prompts


def cluster_prompts(
    prompt_embeddings: np.ndarray,
    n_clusters: int,
    seed: int = 17,
) -> np.ndarray:
    """KMeans over SBERT embeddings. Returns cluster IDs of shape (N,).

    If n_clusters <= 1, returns all-zeros (single global cluster) —
    used for smoke tests where per-cluster archive isn't needed.
    """
    if n_clusters <= 1:
        return np.zeros(prompt_embeddings.shape[0], dtype=int)
    if n_clusters > prompt_embeddings.shape[0]:
        raise ValueError(
            f"FATAL: n_clusters={n_clusters} > n_prompts={prompt_embeddings.shape[0]}"
        )
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(prompt_embeddings)
