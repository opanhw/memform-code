"""Public evaluation protocol.

Framework runners (LangChain, LlamaIndex, defense) import shared helpers from
this module instead of importing each other. The implementation currently lives
in ``memform.eval.core``; this facade is the stable import surface.
"""

from memform.eval.core import (
    DEFAULT_FREEFORM_JUDGE_MODEL,
    FIXED_JUDGE_MODEL,
    FIXED_JUDGE_REASONING_EFFORT,
    CacheOnlyMiss,
    CachedChatModel,
    DiskCache,
    aggregate_results,
    assert_freeform_evaluator_contract,
    case_inventory,
    estimated_calls,
    exp_texts,
    format_experiences,
    format_hms,
    generate_p5_random_orders,
    is_freeform_case,
    judge_target_alignment,
    p5_updater_prompts,
    parse_final_label_details,
    run_case,
    task_diagnostic_record,
    task_prompt,
    utc_now,
    _median,
)

__all__ = [
    "DEFAULT_FREEFORM_JUDGE_MODEL",
    "FIXED_JUDGE_MODEL",
    "FIXED_JUDGE_REASONING_EFFORT",
    "CacheOnlyMiss",
    "CachedChatModel",
    "DiskCache",
    "aggregate_results",
    "assert_freeform_evaluator_contract",
    "case_inventory",
    "estimated_calls",
    "exp_texts",
    "format_experiences",
    "format_hms",
    "generate_p5_random_orders",
    "is_freeform_case",
    "judge_target_alignment",
    "p5_updater_prompts",
    "parse_final_label_details",
    "run_case",
    "task_diagnostic_record",
    "task_prompt",
    "utc_now",
    "_median",
]
