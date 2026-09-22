"""Online memory defenses for MemForm. Each defense lives in its own module."""

from .a_memguard import AMemGuard
from .keyword_filter import KeywordFilter
from .llm_memory_judge import LLMMemoryJudge
from .memory_sanitization import MemorySanitization
from .task_shield import TaskShield

DEFENSE_REGISTRY = {
    "llm_judge": LLMMemoryJudge,
    "memory_sanitization": MemorySanitization,
    "a_memguard": AMemGuard,
    "keyword_filter": KeywordFilter,
    "task_shield": TaskShield,
}

__all__ = [
    "AMemGuard",
    "DEFENSE_REGISTRY",
    "KeywordFilter",
    "LLMMemoryJudge",
    "MemorySanitization",
    "TaskShield",
]
