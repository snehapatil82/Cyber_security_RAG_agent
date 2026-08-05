"""
core/confidence.py — Confidence scoring for generated answers.

Confidence is derived from retrieval evidence quality, not from asking the
LLM to self-report a number (which is unreliable). It combines:
  - mean semantic similarity of retrieved chunks
  - number of retrieved chunks
  - diversity of trusted sources (NVD, CISA KEV, vendor advisory, uploaded report)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfidenceResult:
    score: int  # 0-100
    label: str  # High | Medium | Low
    num_sources: int
    source_list: list[str]


def compute_confidence(retrieved: list[dict], has_live_cve_data: bool = False) -> ConfidenceResult:
    if not retrieved and not has_live_cve_data:
        return ConfidenceResult(score=0, label="Low", num_sources=0, source_list=[])

    similarities = [r.get("similarity", 0.0) for r in retrieved] or [0.0]
    mean_sim = sum(similarities) / len(similarities)

    sources = set()
    for r in retrieved:
        src = r.get("metadata", {}).get("source") or r.get("metadata", {}).get("cve_id") or "unknown"
        sources.add(src)
    if has_live_cve_data:
        sources.add("NVD/CISA")

    volume_factor = min(1.0, len(retrieved) / 4)  # 4+ chunks = full credit
    diversity_bonus = min(0.15, 0.05 * len(sources))

    raw = (mean_sim * 0.7) + (volume_factor * 0.15) + diversity_bonus
    score = max(0, min(100, round(raw * 100)))

    if score >= 75:
        label = "High"
    elif score >= 45:
        label = "Medium"
    else:
        label = "Low"

    return ConfidenceResult(score=score, label=label, num_sources=len(sources), source_list=sorted(sources))
