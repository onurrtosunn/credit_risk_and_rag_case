#!/usr/bin/env python3
"""
Utility functions shared across the project.
Consolidates common functions to avoid code duplication.
"""
import os
import re
from typing import List, Dict, Optional
import functools

try:
    from zeyrek import MorphAnalyzer
    _TR_LEMMA = MorphAnalyzer()
except Exception:
    _TR_LEMMA = None


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    if path:
        os.makedirs(path, exist_ok=True)


@functools.lru_cache(maxsize=200_000)
def _lemma_tr(tok: str) -> str:
    """Lemmatize Turkish token."""
    if not _TR_LEMMA:
        return tok
    try:
        lem = _TR_LEMMA.lemmatize(tok)
        if isinstance(lem, list) and len(lem) > 0:
            cand = lem[0][0] if isinstance(lem[0], (list, tuple)) and len(lem[0]) > 0 else lem[0]
            return str(cand)
    except Exception:
        pass
    try:
        analyses = _TR_LEMMA.analyze(tok)
        if analyses:
            first = analyses[0]
            cand = getattr(first, "lemma", None)
            if cand:
                return str(cand)
    except Exception:
        pass
    return tok


def tr_tokenize(text: str, use_lemmatization: bool = True) -> List[str]:
    """
    Tokenize Turkish text.
    
    Args:
        text: Input text
        use_lemmatization: Whether to use lemmatization (default: True)
    
    Returns:
        List of tokens
    """
    text = text.lower()
    text = re.sub(r"[^\wçğıöşü\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
    toks = text.split() if text else []
    if use_lemmatization:
        return [_lemma_tr(t) for t in toks]
    return toks

