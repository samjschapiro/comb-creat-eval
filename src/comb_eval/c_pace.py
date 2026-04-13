"""Constrained PACE (C-PACE).

Injects lexical constraints into PACE's stage-2 prompt. A chain is "valid"
iff it satisfies every constraint; the diversity signal (PACE's internal
chain score over FastText embeddings) is only aggregated over valid chains.
Constraint satisfaction rate is reported separately — never folded into the
diversity signal — so capability and creativity stay decoupled, matching the
comb_eval framework.

Difficulty levels (mirrors comb_eval):
    L1: no constraints (vanilla PACE chain)
    L2: 1 inclusion OR 1 exclusion (p=0.5)
    L3: 1 inclusion + 1 exclusion
    L4: 2 inclusions + 2 exclusions

Constraints are *lexical*, on the first letter of each word:
- Inclusion letters drawn from common English initial letters (avoids
  generating unsatisfiable constraints on seeds where the associative space
  can't meet a rare letter).
- Exclusion letters drawn from middle-rare initial letters (restrictive but
  not so dominant that chains become trivially unsatisfiable).

Cross-track import note: `FastTextEmbeddings` and `score_chain` are imported
from `src.dat_eval.pace` because PACE's scoring primitives are the exact
thing we're extending — duplicating them would drift. This is the one
sanctioned cross-track dependency in this module.
"""

import json
import random
import re
from dataclasses import dataclass, field, asdict

import numpy as np

from src.dat_eval.pace import (
    FastTextEmbeddings,  # re-exported for the runner/scorer
    score_chain,
    DEFAULT_SEEDS,
)


# Letters for constraint draws. See module docstring.
COMMON_INIT_LETTERS = ("a", "e", "i", "o", "s", "t", "n", "r", "l", "c")
MIDRARE_INIT_LETTERS = ("b", "d", "f", "g", "h", "m", "p", "u", "w")


@dataclass
class Constraints:
    """Lexical constraints on an associative chain."""

    include_letters: list[str] = field(default_factory=list)
    exclude_letters: list[str] = field(default_factory=list)
    level: int = 1

    def is_empty(self) -> bool:
        return not self.include_letters and not self.exclude_letters

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Constraints":
        return cls(
            include_letters=list(d.get("include_letters", [])),
            exclude_letters=list(d.get("exclude_letters", [])),
            level=int(d.get("level", 1)),
        )


def generate_constraints(level: int, rng: random.Random) -> Constraints:
    """Draw constraints for a given difficulty level.

    Inclusion and exclusion letter sets are disjoint to avoid contradictions.
    """
    if level == 1:
        return Constraints(level=1)

    if level == 2:
        # One constraint: include or exclude, 50/50.
        if rng.random() < 0.5:
            letter = rng.choice(COMMON_INIT_LETTERS)
            return Constraints(include_letters=[letter], level=2)
        else:
            letter = rng.choice(MIDRARE_INIT_LETTERS)
            return Constraints(exclude_letters=[letter], level=2)

    if level == 3:
        inc = rng.choice(COMMON_INIT_LETTERS)
        # exclude from midrare AND not equal to inc (already disjoint by construction)
        exc = rng.choice(MIDRARE_INIT_LETTERS)
        return Constraints(include_letters=[inc], exclude_letters=[exc], level=3)

    if level == 4:
        inc = rng.sample(COMMON_INIT_LETTERS, 2)
        exc = rng.sample(MIDRARE_INIT_LETTERS, 2)
        return Constraints(include_letters=sorted(inc), exclude_letters=sorted(exc), level=4)

    raise ValueError(f"Unsupported level: {level}")


# --- prompting ---


def c_pace_stage1_prompt(seed: str) -> str:
    """Stage 1: seed → 3 first-associations. Identical to vanilla PACE."""
    return (
        f'Starting with the word "{seed}", generate three different words that '
        f"directly associate with this initial word only (not with each other). "
        f"Please put down only single words, and do not use proper nouns "
        f"(such as names, brands, etc.). "
        f"For each word, provide a brief explanation of its connection to "
        f'"{seed}". Return in JSON format:\n'
        f'{{"results": [{{"word": "", "reason": ""}}, '
        f'{{"word": "", "reason": ""}}, '
        f'{{"word": "", "reason": ""}}]}}'
    )


def c_pace_stage2_prompt(
    seed: str,
    second_word: str,
    reason: str,
    constraints: Constraints,
) -> str:
    """Stage 2: constrained 20-word chain.

    When constraints is empty (L1) this is identical to vanilla PACE. For
    L2-L4 we prepend explicit instruction block about lexical constraints
    on the *first letter* of each word.
    """
    base = (
        f'Starting with the word pair "{seed}" -> "{second_word}", generate a '
        f"chain of 20 words where each new word should be associated with ONLY "
        f"the word immediately before it. Generate the third word based on "
        f'"{second_word}", then generate the fourth word based on your third '
        f"word, and so on. Please put down only single words, and do not use "
        f"proper nouns (such as names, brands, etc.)."
    )

    if not constraints.is_empty():
        rules = ["\n\nADDITIONAL CONSTRAINTS on the 20-word chain:"]
        if constraints.include_letters:
            letters = ", ".join(f"'{l}'" for l in constraints.include_letters)
            rules.append(
                f"- The chain MUST include at least one word starting with each "
                f"of these letters (case-insensitive): {letters}."
            )
        if constraints.exclude_letters:
            letters = ", ".join(f"'{l}'" for l in constraints.exclude_letters)
            rules.append(
                f"- The chain must NOT include any word starting with these "
                f"letters (case-insensitive): {letters}."
            )
        rules.append(
            "Satisfy every constraint while still producing associative chains "
            "where each word connects to the one before it."
        )
        base = base + "\n".join(rules)

    base += (
        f"\n\nFor each word, provide a brief explanation of its connection to "
        f"the previous word. Return in JSON format with exactly 20 entries:\n"
        f'{{"results": [{{"word": "{second_word}", "reason": "{reason}"}}, '
        f'{{"word": "", "reason": ""}}, ... ]}}'
    )
    return base


# --- parsing ---


def parse_stage1(raw: str | None) -> list[dict]:
    """Parse stage-1 JSON into [{word, reason}, ...] (up to 3)."""
    if not raw:
        return []
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group())
        results = data.get("results", [])
        if not isinstance(results, list):
            return []
        return [
            {
                "word": str(r.get("word", "")).strip(),
                "reason": str(r.get("reason", "")).strip(),
            }
            for r in results[:3]
            if isinstance(r, dict) and r.get("word")
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def parse_stage2_chain(raw: str | None, seed: str, second_word: str) -> list[str]:
    """Parse stage-2 chain JSON into a list of word strings (lowercased).

    Returns [seed, second_word, w3, w4, ...] up to 20 entries. Falls back to
    a best-effort regex extraction if JSON fails.
    """
    chain = [seed.lower(), second_word.lower()]
    if not raw:
        return chain

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            results = data.get("results", [])
            if isinstance(results, list):
                for entry in results:
                    if not isinstance(entry, dict):
                        continue
                    word = str(entry.get("word", "")).strip().lower()
                    if word:
                        chain.append(word)
                return chain[:20]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: scrape single-word tokens from the response
    # Keep only alphabetic tokens to avoid picking up numbers/indices
    tokens = re.findall(r"\b[a-zA-Z]+\b", raw)
    for t in tokens:
        tl = t.lower()
        if tl not in {"results", "word", "reason"}:
            chain.append(tl)
        if len(chain) >= 20:
            break
    return chain[:20]


# --- constraint verification ---


def check_constraints(chain: list[str], constraints: Constraints) -> dict:
    """Verify a chain against its constraints. Returns per-constraint booleans.

    The chain's first word is the seed (given, not the model's choice). We
    evaluate constraints over all words except the seed, which is the space
    the model actually controls.
    """
    # Exclude the seed from constraint checks — the model didn't choose it.
    chain_body = chain[1:]
    chain_initials = {w[0].lower() for w in chain_body if w}

    include_results = {
        l: (l.lower() in chain_initials)
        for l in constraints.include_letters
    }
    exclude_results = {
        l: (l.lower() not in chain_initials)
        for l in constraints.exclude_letters
    }

    satisfied = all(include_results.values()) and all(exclude_results.values())

    return {
        "satisfied": satisfied,
        "include_results": include_results,
        "exclude_results": exclude_results,
    }


# --- aggregate scoring ---


def compute_utility_hard(
    constraints: Constraints,
    satisfied: bool,
    alpha_I: float = 1.0,
    alpha_X: float = 1.0,
) -> float:
    """Hard utility (Schapiro et al., arXiv:2509.21043).

    U_hard(x) = [1 + alpha_I * |I|] * [1 + alpha_X * |X|] * 1[all constraints satisfied]

    Zero on ANY violation (binary gate). Otherwise scales multiplicatively
    with constraint count, rewarding harder problems. With alpha = 1:
        L1 (0I, 0X) -> 1
        L2 (1I or 1X) -> 2
        L3 (1I + 1X) -> 4
        L4 (2I + 2X) -> 9
    """
    if not satisfied:
        return 0.0
    n_I = len(constraints.include_letters)
    n_X = len(constraints.exclude_letters)
    return (1.0 + alpha_I * n_I) * (1.0 + alpha_X * n_X)


def compute_utility_soft(
    constraints: Constraints,
    include_results: dict,
    exclude_results: dict,
) -> float:
    """Soft utility — partial credit for partially-satisfied constraint sets.

    U_soft(x) = n_satisfied / n_total, in [0, 1].

    For L1 (no constraints) there is nothing to satisfy; by convention
    U_soft = 1.0 so L1 composite stays comparable to novelty.
    """
    n_total = len(constraints.include_letters) + len(constraints.exclude_letters)
    if n_total == 0:
        return 1.0
    n_sat = sum(1 for ok in include_results.values() if ok) + \
            sum(1 for ok in exclude_results.values() if ok)
    return n_sat / n_total


@dataclass
class CPaceResult:
    """Per-(seed, first_assoc, level) scoring result."""

    seed: str
    second_word: str
    level: int
    constraints: dict
    chain: list[str]
    chain_score: float  # PACE internal-chain diversity (Novelty, N)
    n_oov: int
    satisfied: bool
    include_results: dict
    exclude_results: dict
    utility_hard: float                  # Schapiro U, binary-gated
    utility_soft: float                  # fraction satisfied, in [0, 1]
    composite_creativity_hard: float     # U_hard * N
    composite_creativity_soft: float     # U_soft * N


def score_one(
    chain: list[str],
    constraints: Constraints,
    embeddings: FastTextEmbeddings,
    seed: str,
    second_word: str,
    alpha_I: float = 1.0,
    alpha_X: float = 1.0,
) -> CPaceResult:
    pace_out = score_chain(chain, embeddings)
    check = check_constraints(chain, constraints)
    satisfied = bool(check["satisfied"])
    novelty = float(pace_out["chain_score"])
    utility_hard = compute_utility_hard(constraints, satisfied, alpha_I, alpha_X)
    utility_soft = compute_utility_soft(
        constraints, check["include_results"], check["exclude_results"],
    )
    return CPaceResult(
        seed=seed,
        second_word=second_word,
        level=constraints.level,
        constraints=constraints.to_dict(),
        chain=pace_out["words"],
        chain_score=novelty,
        n_oov=int(pace_out.get("n_oov", 0)),
        satisfied=satisfied,
        include_results=check["include_results"],
        exclude_results=check["exclude_results"],
        utility_hard=utility_hard,
        utility_soft=utility_soft,
        composite_creativity_hard=utility_hard * novelty,
        composite_creativity_soft=utility_soft * novelty,
    )


def aggregate_by_level(results: list[CPaceResult]) -> dict:
    """Aggregate C-PACE results.

    Three scoring channels reported per level and overall:

    1. `mean_composite_creativity` = mean(U * N) across **all** chains
       (including zero-utility failures). This is the Schapiro et al. headline
       metric. Couples capability and creativity by construction.

    2. `mean_chain_score_valid` = mean(N) across chains that satisfied
       constraints. NaN when n_valid == 0. This is the decoupled creativity
       channel (PACE-style; capability failures drop out, not zero out).

    3. `constraint_satisfaction_rate` = fraction of chains with satisfied
       constraints. Pure capability channel.

    Plus `mean_chain_score_all` as a control (mean N over all chains).
    """
    by_level: dict[int, list[CPaceResult]] = {}
    for r in results:
        by_level.setdefault(r.level, []).append(r)

    def _bucket(rs: list[CPaceResult]) -> dict:
        valid_scores = [r.chain_score for r in rs if r.satisfied and r.chain_score > 0]
        all_scores = [r.chain_score for r in rs if r.chain_score > 0]
        n_valid = sum(1 for r in rs if r.satisfied)
        composites_hard = [r.composite_creativity_hard for r in rs]
        composites_soft = [r.composite_creativity_soft for r in rs]
        utilities_hard = [r.utility_hard for r in rs]
        utilities_soft = [r.utility_soft for r in rs]
        return {
            "n_chains": len(rs),
            "n_valid": n_valid,
            "constraint_satisfaction_rate": n_valid / len(rs) if rs else float("nan"),
            "mean_chain_score_valid": float(np.mean(valid_scores)) if valid_scores else float("nan"),
            "mean_chain_score_all": float(np.mean(all_scores)) if all_scores else float("nan"),
            "mean_utility_hard": float(np.mean(utilities_hard)) if utilities_hard else float("nan"),
            "mean_utility_soft": float(np.mean(utilities_soft)) if utilities_soft else float("nan"),
            "mean_composite_creativity_hard": float(np.mean(composites_hard)) if composites_hard else float("nan"),
            "mean_composite_creativity_soft": float(np.mean(composites_soft)) if composites_soft else float("nan"),
        }

    out: dict = {"by_level": {}}
    for level, rs in sorted(by_level.items()):
        out["by_level"][level] = _bucket(rs)
    out["overall"] = _bucket(results)
    return out
