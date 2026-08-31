"""Per-task answer-string normalizers for the GSM8K-flow pipeline.

EM-style metrics compare the model's decoded answer string to the gold
answer after normalization. Different tasks need different normalizers:

  numeric    -- gsm8k-formulas (default): strip currency/commas/period.
  smiles     -- mech, molpuzzle: RDKit-canonicalize the SMILES (so
                "C(C)C" == "CCC" after normalization).
  set_smiles -- retro starting_materials = "A.B.C" of unordered reactants:
                canonicalize each component, sort, join with ".".
  classify   -- prontoqa / proofwriter ("True" / "False" / "Unknown"):
                lowercase + strip trailing punctuation.

Used by:
  - train_gsm8k_flow.py:run_val_em (val-EM-best ckpt selection)
  - test_gsm8k_flow.py:normalize_answer (final eval)
  - scripts/multi_seed_vote_eval.py (multi-seed inference)

Selection is via a CLI flag --answer_normalizer (default "numeric").
"""
from __future__ import annotations

from typing import Callable, Dict


def _normalize_numeric(s: str) -> str:
    return s.replace(",", "").replace("$", "").rstrip(".").strip()


def _normalize_classify(s: str) -> str:
    s = s.strip().lower().rstrip(".").rstrip(",").strip()
    # Standardize whitespace and pull off any leading "answer:" decoration the
    # answer head sometimes emits.
    if s.startswith("answer:"):
        s = s[len("answer:"):].strip()
    if s.startswith("the answer is"):
        s = s[len("the answer is"):].strip()
    return s


_RDKIT_OK: bool | None = None
_CHEM = None


def _import_rdkit():
    global _RDKIT_OK, _CHEM
    if _RDKIT_OK is None:
        try:
            from rdkit import (
                Chem,  # noqa: F401
                RDLogger,
            )
            RDLogger.DisableLog("rdApp.*")  # silence parse warnings
            _CHEM = Chem
            _RDKIT_OK = True
        except Exception:
            _RDKIT_OK = False
    return _CHEM if _RDKIT_OK else None


def _canonicalize_one(smiles: str) -> str:
    """RDKit-canonicalize a single SMILES.

    If the raw string fails to parse (e.g. trailing garbage from a
    decoder that kept emitting tokens after the answer), we try
    progressively shorter prefixes until one parses, up to 64 char
    trims. This salvages cases where the model emitted the right
    answer followed by junk like " 1) ||| ([1, 101], 101)".

    If nothing parses, return the trimmed input string (strict EM).
    """
    s = (smiles or "").strip()
    if not s:
        return s
    Chem = _import_rdkit()
    if Chem is None:
        return s
    # Fast path: the whole string parses.
    mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)
    # Salvage path: find the longest prefix that parses. We try
    # truncating from the right, both word-by-word (split on common
    # decoder garbage tokens) and char-by-char.
    candidates = []
    for sep in (" ||| ", " ", "\t"):
        if sep in s:
            candidates.append(s.split(sep, 1)[0])
    candidates.append(s)
    for cand in candidates:
        cand = cand.strip()
        # Try the candidate as-is.
        mol = Chem.MolFromSmiles(cand)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        # Try trimming up to 32 trailing characters. We rstrip only
        # punctuation that's never part of valid SMILES at the end:
        # parentheses (unmatched), comma, period (only if dangling),
        # whitespace. We do NOT strip square brackets since those are
        # part of every atom-mapped SMILES.
        for k in range(1, min(33, len(cand))):
            trimmed = cand[:-k].rstrip("(),. \t\r\n")
            if not trimmed:
                continue
            mol = Chem.MolFromSmiles(trimmed)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
    return s


def _normalize_smiles(s: str) -> str:
    return _canonicalize_one(s)


def _normalize_set_smiles(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    parts = [_canonicalize_one(p) for p in s.split(".") if p.strip()]
    parts = [p for p in parts if p]
    return ".".join(sorted(parts))


_NORMALIZERS: Dict[str, Callable[[str], str]] = {
    "numeric":    _normalize_numeric,
    "smiles":     _normalize_smiles,
    "set_smiles": _normalize_set_smiles,
    "classify":   _normalize_classify,
}


VALID_NORMALIZERS = sorted(_NORMALIZERS.keys())


def get_normalizer(name: str) -> Callable[[str], str]:
    """Return the answer-string normalizer for the given name. Raises a
    ValueError on unknown name (so a typo in --answer_normalizer fails
    fast rather than silently using a wrong metric)."""
    if name not in _NORMALIZERS:
        raise ValueError(
            f"unknown --answer_normalizer={name!r}; "
            f"valid: {VALID_NORMALIZERS}"
        )
    return _NORMALIZERS[name]
