"""Scoring for the anagram (exploratory-creativity) task -- judge-free by design.

An anagram candidate is scored on two objective *utility* floors, then novelty:
  1. is_valid_anagram -- uses EXACTLY the source's letters (a multiset match), rearranged. This is
     the hard, deterministic floor; it catches the character-level hallucinations LLMs produce
     (extra/missing/wrong-count letters) that merely *look* like anagrams.
  2. is_meaningful -- every whitespace-separated token is a real English word (a single word =>
     that one word). Uses a spell-checker lexicon (pyspellchecker), NOT raw frequency and NOT the
     archaic macOS wordlist: frequency conflates rare-real words with common junk (e.g. 'tinsel'
     4.5e-7 is *rarer* than the junk fragment 'kan' 2.3e-6, so no frequency floor separates them),
     and the vintage /usr/share/dict lacks inflections ('desserts') while including archaic junk.
     The spell lexicon recognises inflections and rare-but-real words while rejecting fragments.
Novelty = embedding distance between the source and the candidate (scoring.cosine_distance) -- the
SAME remoteness measure as the combinatorial tasks; a distant valid anagram is the creative goal.
No LLM judge anywhere: the anagram task is the framework's fully-objective endpoint.
"""

from __future__ import annotations

import re

from spellchecker import SpellChecker

_SPELL = SpellChecker()   # curated English lexicon; handles inflections + rejects junk fragments


def _letters(s: str) -> list[str]:
    return sorted(c for c in s.lower() if c.isalpha())


def is_valid_anagram(source: str, candidate: str) -> bool:
    """True iff `candidate` uses exactly `source`'s letters (multiset), in a different arrangement.

    Case-insensitive; ignores spaces/punctuation/digits (standard anagram convention). This is a
    MULTISET test (counts matter), so 'London'->'no doll' fails: both use {d,l,n,o} but the counts
    differ (London has NN+OO; 'no doll' has LL+N).
    """
    src = _letters(source)
    if not src or _letters(candidate) != src:
        return False
    return source.lower().replace(" ", "") != candidate.lower().replace(" ", "")


def is_meaningful(candidate: str, entity_check=None) -> bool:
    """True iff `candidate` is a real word/phrase, or (optionally) a real named entity.

    Every whitespace token must be a known lexicon word (a single word => that word). As a fallback
    for proper-noun anagrams the lexicon lacks (a brand, place, or name), the WHOLE candidate may
    instead be a real entity: pass ``entity_check=wikidata.is_entity_label`` (opt-in; makes a cached,
    paced network call). Default ``None`` keeps the scorer fully offline/free.
    """
    toks = [re.sub(r"[^a-z]", "", t.lower()) for t in candidate.split()]
    toks = [t for t in toks if t]
    if toks and all(_SPELL.known([t]) for t in toks):
        return True
    return bool(entity_check) and bool(entity_check(candidate.strip()))


def score_candidate(source: str, candidate: str, embed=None, entity_check=None) -> dict:
    """Full judge-free score for one candidate: validity, meaningfulness, and (optional) novelty."""
    valid = is_valid_anagram(source, candidate)
    meaningful = is_meaningful(candidate, entity_check=entity_check) if valid else False
    out = {"candidate": candidate, "valid_anagram": valid, "meaningful": meaningful,
           "utility": valid and meaningful}
    if embed is not None and valid:
        from src.kg_creat.scoring import cosine_distance
        out["novelty"] = cosine_distance(embed(source), embed(candidate))
    return out
