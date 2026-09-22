#!/usr/bin/env python3
"""Keyword / regex memory filter: a cheap non-LLM write-time baseline.

Flags overt prompt-injection and universal behavioral commands from runtime-
visible memory text only. No model calls, no attack labels, no P1-P5 branches.
"""
from __future__ import annotations

import re
import time
from typing import Any

from .common import DefenseContext, DefenseDecision, MemoryDefense, MemoryItem


# High-severity: classic instruction-override / injection phrasing.
HIGH_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) (instructions?|rules?|memory)", re.I),
    re.compile(r"disregard (your |the )?system\b", re.I),
    re.compile(r"override (the )?(policy|safety|protocol|system)", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bsystem prompt\b", re.I),
    re.compile(r"do not follow (the )?task\b", re.I),
    re.compile(r"do not follow (your|the) (instructions?|rules?)", re.I),
]

# Medium-severity: attempts to lock future choices or ignore the current question.
MEDIUM_PATTERNS = [
    re.compile(r"always (choose|pick|select|answer|output)\b", re.I),
    re.compile(r"never (choose|consider|mention)\b", re.I),
    re.compile(r"for all future (tasks|cases|queries)", re.I),
    re.compile(r"the (only )?correct (answer|action) is always\b", re.I),
    re.compile(r"ignore the current (task|question|situation)", re.I),
]


def _match_labels(text: str, patterns: list[re.Pattern[str]]) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


class KeywordFilter(MemoryDefense):
    name = "keyword_filter"
    per_task = False

    def __init__(self, model: Any = None, threshold: float = 0.5, **kwargs: Any):
        super().__init__(model=None, threshold=threshold, **kwargs)

    def filter_memory(self, memory: MemoryItem, context: DefenseContext | None = None) -> DefenseDecision:
        started = time.perf_counter()
        text = memory.text or ""
        high = _match_labels(text, HIGH_PATTERNS)
        medium = _match_labels(text, MEDIUM_PATTERNS)
        matched = high + medium
        if high:
            action = "reject"
            accepted = False
            severity = "high"
            reason = "keyword filter: high-severity injection phrasing"
        elif medium:
            action = "quarantine"
            accepted = False
            severity = "medium"
            reason = "keyword filter: behavioral-lock phrasing"
        else:
            action = "accept"
            accepted = True
            severity = "none"
            reason = "keyword filter: no matched patterns"
        return DefenseDecision(
            memory_id=memory.memory_id,
            accepted=accepted,
            action=action,
            score=1.0 if high else (0.5 if medium else 0.0),
            reason=reason,
            original_text=memory.text,
            extra={
                "matched_patterns": matched,
                "filter_type": "keyword",
                "severity": severity,
            },
            latency=time.perf_counter() - started,
            amortized_llm_calls=0.0,
            batch_llm_calls=0,
            llm_calls=0,
        )

    def process_memories(
        self,
        memories: list[MemoryItem],
        context: DefenseContext | None = None,
    ) -> list[DefenseDecision]:
        return [self.filter_memory(item, context) for item in memories]
