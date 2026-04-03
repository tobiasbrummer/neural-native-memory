"""Prompt utilities for KV-Embedding."""

from __future__ import annotations

from typing import List, Sequence

from .config import PromptRole

ROLE_LABELS = {
    "context": "Context",
    "query": "Query",
}


def build_compression_prompt(
    text: str,
    role: PromptRole,
    template: str,
) -> str:
    label = ROLE_LABELS[role]
    return template.format(role=label, text=text)


def build_compression_prompts(
    texts: Sequence[str],
    roles: Sequence[PromptRole],
    template: str,
) -> List[str]:
    if len(texts) != len(roles):
        raise ValueError("texts and roles must have the same length")
    return [build_compression_prompt(t, r, template) for t, r in zip(texts, roles)]
