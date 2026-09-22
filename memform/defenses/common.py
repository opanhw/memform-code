#!/usr/bin/env python3
"""Shared defense interface and MemForm adapters.

Ground-truth attack fields never enter a defense. The adapter may read
`case["interface"]` only to locate history fields in the compiled JSON; that value
is not passed into any defense model prompt.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


EMPTY_MEMORY = "No reusable decision rule has been learned yet."
DEFENSE_BLOCKED_SENTINEL = "[[DEFENSE_OUTPUT_BLOCKED]]"
TECHNICAL_FAILURE_CLASSES = frozenset({"parse_error", "tu_parse_failure"})

# Dict keys that must never be used as detection features or shown to a defense model.
LEAK_FIELDS = frozenset({
    "attack_interface", "attack_label", "attack_type", "attack_annotation",
    "poisoned", "clean", "ground_truth", "target_behavior", "expected_answer",
    "supports", "scope_bucket", "provenance", "authentic", "origin",
    "contains_forged_evidence", "clean_policy", "poisoned_policy",
    "clean_answer", "poisoned_answer", "oracle_memory", "interface",
    "poison_target_hit", "causal_attack_success", "family_id", "case_id",
    "condition",
})

# Labels that must not appear in the prompt WRAPPER (after memory text is stripped).
# Do not scan the memory's own natural-language content for these words.
LEAK_WRAPPER_LABELS = (
    "attack_interface",
    "attack_label",
    "attack_type",
    "ground_truth",
    "target_behavior",
    "expected_answer",
    "family_id",
    "case_id",
    "memory_id",
    "learned_clean",
    "learned_poisoned",
    "poison_target",
)


class ChatModelLike(Protocol):
    def invoke(self, system: str, user: str) -> str: ...


@dataclass
class MemoryItem:
    """Runtime-visible memory/experience. No attack labels."""

    memory_id: str
    text: str
    context: str | None = None
    action: str | None = None
    outcome: str | None = None


@dataclass
class DefenseContext:
    """What a real agent/memory system could see. No gold answers or interface tags."""

    decision_variable: str
    domain: str | None = None
    task_text: str | None = None
    choices: list[dict[str, str]] | None = None


@dataclass
class AgentRuntimeInput:
    """Runtime-visible generation context for use-time defenses. No evaluation labels."""

    context: DefenseContext
    memory_text: str | None = None
    task_format: str = "mcq"
    choice_map: dict[str, str] | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class DefenseDecision:
    memory_id: str
    accepted: bool
    action: str  # keep | reject | sanitize | accept | quarantine | bypass
    score: float | None = None
    reason: str = ""
    sanitized_text: str | None = None
    original_text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    latency: float = 0.0
    llm_calls: int = 0  # deprecated alias of batch size share; prefer amortized_llm_calls
    amortized_llm_calls: float = 0.0
    batch_llm_calls: int = 0

    def to_log_dict(self) -> dict[str, Any]:
        row = {
            "memory_id": self.memory_id,
            "accepted": self.accepted,
            "action": self.action,
            "decision": self.action if self.action in {
                "accept", "reject", "quarantine", "keep", "sanitize", "bypass",
                "parse_failure_bypass", "feedback", "block",
            } else ("accept" if self.accepted else "reject"),
            "score": self.score,
            "reason": self.reason,
            "original_memory": self.original_text,
            "sanitized_memory": self.sanitized_text,
            "latency": round(self.latency, 4),
            "amortized_llm_calls": round(self.amortized_llm_calls, 4),
            "batch_llm_calls": int(self.batch_llm_calls),
            "llm_calls": round(self.amortized_llm_calls, 4),
        }
        row.update(self.extra)
        return row


class MemoryDefense:
    """Online memory defense. Subclasses must not read attack labels."""

    name: str = "base"
    display_name: str = ""
    per_task: bool = False
    applies_write_time: bool = True
    intercepts_agent_generation: bool = False
    prompt_version: str = "v1"

    def __init__(self, model: ChatModelLike | None = None, threshold: float = 0.5, **kwargs: Any):
        self.model = model
        self.threshold = float(threshold)
        self.extra_kwargs = kwargs
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._excluded_texts: list[str] = []

    def begin_case(self, case_id: str) -> None:
        """Reset per-case state. `case_id` is bookkeeping only, never a prompt feature."""
        self.reset_state()

    def reset_state(self) -> None:
        """Clear internal defense state. Does not take benchmark condition names."""
        return None

    def export_state(self) -> dict[str, Any]:
        return {}

    def import_state(self, state: dict[str, Any] | None) -> None:
        return None

    def filter_memory(self, memory: MemoryItem, context: DefenseContext | None = None) -> DefenseDecision:
        raise NotImplementedError

    def process_memories(
        self,
        memories: list[MemoryItem],
        context: DefenseContext | None = None,
    ) -> list[DefenseDecision]:
        return [self.filter_memory(m, context) for m in memories]

    def prepare_agent_memory(
        self,
        memory_text: str | None,
        context: DefenseContext | None = None,
        condition: str | None = None,
    ) -> str | None:
        # `condition` is accepted for caller compatibility and ignored as a feature.
        return memory_text

    def intercept_agent_generation(
        self,
        generate: Callable[[str, str], str],
        system: str,
        user: str,
        runtime: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Optional use-time hook. Default: generate once, no feedback retry."""
        return generate(system, user), {}

    def invoke_llm(self, system: str, user: str, *, excluded_texts: list[str] | None = None) -> str:
        if self.model is None:
            raise RuntimeError(f"{self.name} requires a chat model")
        texts = list(excluded_texts if excluded_texts is not None else self._excluded_texts)
        assert_prompt_no_metadata_leak(system, user, excluded_texts=texts)
        self.llm_calls += 1
        return self.model.invoke(system, user)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.S)
    if fenced:
        candidates.append(fenced.group(1))
    m = re.search(r"\{.*\}", stripped, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def extract_json_array(text: str) -> list[Any] | None:
    stripped = str(text or "").strip()
    obj = extract_json_object(stripped)
    if isinstance(obj, dict):
        for key in ("decisions", "items", "results", "paths", "judgments"):
            if isinstance(obj.get(key), list):
                return obj[key]
    m = re.search(r"\[.*\]", stripped, flags=re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    return arr if isinstance(arr, list) else None


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    return max(0.0, min(1.0, x))


def experience_to_item(exp: dict[str, Any]) -> MemoryItem:
    """Strip compiler/evaluator fields before a defense sees the record."""
    text = str(exp.get("text") or "").strip()
    if not text:
        parts = [
            str(exp.get("context") or "").strip(),
            str(exp.get("action") or "").strip(),
            str(exp.get("outcome") or "").strip(),
        ]
        text = " ".join(p for p in parts if p)
    return MemoryItem(
        memory_id=str(exp.get("id") or "").strip(),
        text=text,
        context=str(exp.get("context") or "").strip() or None,
        action=str(exp.get("action") or "").strip() or None,
        outcome=str(exp.get("outcome") or "").strip() or None,
    )


def assert_no_leakage(payload: Any, where: str) -> None:
    """Fail loudly if a defense prompt/context still contains GT keys."""
    if isinstance(payload, dict):
        leaked = sorted(k for k in payload.keys() if k in LEAK_FIELDS)
        if leaked:
            raise RuntimeError(f"ground-truth leak in {where}: {leaked}")
        for v in payload.values():
            assert_no_leakage(v, where)
    elif isinstance(payload, list):
        for v in payload:
            assert_no_leakage(v, where)


def _strip_excluded(text: str, excluded_texts: list[str]) -> str:
    remainder = text
    # Strip longer blobs first so nested fragments are removed cleanly.
    for blob in sorted((t for t in excluded_texts if t), key=len, reverse=True):
        remainder = remainder.replace(blob, "")
    return remainder


def assert_prompt_no_metadata_leak(
    system: str,
    user: str,
    *,
    excluded_texts: list[str] | None = None,
) -> None:
    """Check that benchmark metadata is not spliced into the defense prompt wrapper.

    Memory body text is excluded: a legitimate record may contain words like
    "poisoned" in natural language. We only inspect the surrounding template.
    """
    wrapper = _strip_excluded(f"{system}\n{user}", list(excluded_texts or []))
    lowered = wrapper.lower()
    for label in LEAK_WRAPPER_LABELS:
        if label.lower() in lowered:
            raise RuntimeError(f"defense prompt wrapper contains metadata label {label!r}")
    if re.search(r":p[1-5]:(clean|poisoned)", lowered):
        raise RuntimeError("defense prompt wrapper contains condition-tagged memory id")
    if re.search(r"\b(clean|poisoned)\s+(condition|label|memory|history)\b", lowered):
        raise RuntimeError("defense prompt wrapper names an evaluation condition")
    if re.search(r"\bcandidate memory id\b", lowered):
        raise RuntimeError("defense prompt wrapper includes memory_id")
    # Template-only checks. Runtime-visible memory/task text is stripped first.
    if re.search(r"\b(p[1-5])\b", lowered):
        raise RuntimeError("defense prompt wrapper names an attack interface")
    if re.search(r"\battack\b", lowered):
        raise RuntimeError("defense prompt wrapper names an attack")
    if re.search(r"\bground[ _-]?truth\b", lowered):
        raise RuntimeError("defense prompt wrapper names ground truth")


def defense_context_from_case(
    case: dict[str, Any],
    task: dict[str, Any] | None = None,
) -> DefenseContext:
    choices = None
    task_text = None
    if task:
        task_text = str(task.get("task") or "").strip() or None
        raw_choices = task.get("choices") or []
        if isinstance(raw_choices, list) and raw_choices:
            choices = [
                {"label": str(x.get("label", "")).strip(), "text": str(x.get("text", "")).strip()}
                for x in raw_choices
                if isinstance(x, dict)
            ]
    ctx = DefenseContext(
        decision_variable=str(case.get("decision_variable") or "").strip(),
        domain=str(case.get("domain") or "").strip() or None,
        task_text=task_text,
        choices=choices,
    )
    assert_no_leakage(ctx.__dict__, "DefenseContext")
    return ctx


def runtime_visible_texts(
    items: MemoryItem | list[MemoryItem] | None = None,
    context: DefenseContext | None = None,
) -> list[str]:
    """Natural-language blobs that may legally contain words like 'poisoned'."""
    texts: list[str] = []
    seq: list[MemoryItem]
    if items is None:
        seq = []
    elif isinstance(items, MemoryItem):
        seq = [items]
    else:
        seq = list(items)
    for item in seq:
        for val in (item.text, item.context, item.action, item.outcome):
            if val:
                texts.append(str(val))
    if context is not None:
        for val in (context.decision_variable, context.domain, context.task_text):
            if val:
                texts.append(str(val))
        for choice in context.choices or []:
            text = str(choice.get("text") or "").strip()
            if text:
                texts.append(text)
    return texts


def format_context_block(context: DefenseContext | None) -> str:
    if context is None:
        return ""
    lines = []
    if context.decision_variable:
        lines.append(f"Decision variable: {context.decision_variable}")
    if context.domain:
        lines.append(f"Operational domain: {context.domain}")
    if context.task_text:
        lines.append(f"Current task:\n{context.task_text}")
    if context.choices:
        choice_lines = "\n".join(f"{c['label']}. {c['text']}" for c in context.choices if c.get("label"))
        if choice_lines:
            lines.append(f"Available actions:\n{choice_lines}")
    return "\n".join(lines)


def render_indexed_candidates(items: list[MemoryItem]) -> str:
    """Anonymous 1-based candidate blocks. memory_id stays in Python only."""
    blocks = []
    for i, item in enumerate(items, 1):
        blocks.append(f"Candidate {i}:\n{item.text}")
    return "\n\n".join(blocks)


def apply_amortized_calls(decisions: list[DefenseDecision], batch_calls: int, elapsed: float) -> None:
    n = max(1, len(decisions))
    share = float(batch_calls) / n
    per_latency = elapsed / n
    for dec in decisions:
        dec.batch_llm_calls = int(batch_calls)
        dec.amortized_llm_calls = share
        # Do not store the batch size on llm_calls; case totals must use defense.llm_calls.
        dec.llm_calls = 0
        if not dec.latency:
            dec.latency = per_latency


def apply_stage_overhead(decision: DefenseDecision, overhead: dict[str, Any] | None) -> None:
    """Assign LLM calls/latency to one decision log instead of splitting a later delta."""
    payload = overhead or {}
    calls = int(payload.get("llm_calls") or 0)
    latency = float(payload.get("latency_seconds") or 0.0)
    decision.batch_llm_calls = calls
    decision.amortized_llm_calls = float(calls)
    decision.llm_calls = 0
    decision.latency = latency


def _exp_by_id(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(x.get("id", "")): x for x in case.get("candidate_experiences") or [] if isinstance(x, dict)}


def items_from_ids(case: dict[str, Any], ids: list[str]) -> list[MemoryItem]:
    mapping = _exp_by_id(case)
    items: list[MemoryItem] = []
    for eid in ids:
        exp = mapping.get(str(eid))
        if exp is None:
            continue
        items.append(experience_to_item(exp))
    return items


def primary_history_ids(case: dict[str, Any], condition: str) -> tuple[str, list[str]]:
    """Locate history fields in compiled JSON. `condition` is adapter-only, never a model feature."""
    interface = str(case.get("interface") or "")
    mb = case.get("memory_building") or {}
    block = mb.get(f"{condition}_condition") or {}
    if interface == "P1":
        return "memory_text", []
    if interface == "P2":
        return "evidence_ids", list(block.get("evidence_ids") or [])
    if interface == "P3":
        return "experience_ids", list(block.get("experience_ids") or [])
    if interface == "P4":
        # Fairness: the attack is the retained subset. Never substitute full_pool here.
        ids = list(block.get("order") or block.get("selected_ids") or [])
        return "selected_ids", ids
    if interface == "P5":
        return "order", list(block.get("order") or [])
    return "experience_ids", []


def control_histories(case: dict[str, Any]) -> dict[str, tuple[str, list[str]]]:
    mb = case.get("memory_building") or {}
    controls = mb.get("mechanism_controls") or {}
    out: dict[str, tuple[str, list[str]]] = {}
    for name, block in controls.items():
        if not isinstance(block, dict):
            continue
        if "experience_ids" in block:
            out[name] = ("experience_ids", list(block.get("experience_ids") or []))
        elif "order" in block:
            out[name] = ("order", list(block.get("order") or []))
        elif "selected_ids" in block:
            out[name] = ("selected_ids", list(block.get("selected_ids") or []))
    return out


def _write_history_ids(case: dict[str, Any], condition: str, kind: str, ids: list[str]) -> None:
    mb = case.setdefault("memory_building", {})
    block = mb.setdefault(f"{condition}_condition", {})
    if kind == "evidence_ids":
        block["evidence_ids"] = list(ids)
        block["evidence_budget"] = len(ids)
    elif kind == "experience_ids":
        block["experience_ids"] = list(ids)
        block["experience_budget"] = len(ids)
    elif kind == "selected_ids":
        block["selected_ids"] = list(ids)
        block["order"] = list(ids)
    elif kind == "order":
        block["order"] = list(ids)


def _apply_text_overrides(case: dict[str, Any], overrides: dict[str, str]) -> None:
    if not overrides:
        return
    for exp in case.get("candidate_experiences") or []:
        eid = str(exp.get("id", ""))
        if eid in overrides:
            exp["text"] = overrides[eid]


def decisions_to_kept(
    items: list[MemoryItem],
    decisions: list[DefenseDecision],
) -> tuple[list[str], dict[str, str], list[MemoryItem]]:
    """Keep original relative order. Drop rejects. Apply sanitized text."""
    by_id = {d.memory_id: d for d in decisions}
    kept_ids: list[str] = []
    overrides: dict[str, str] = {}
    kept_items: list[MemoryItem] = []
    for item in items:
        dec = by_id.get(item.memory_id)
        if dec is None or not dec.accepted:
            continue
        new_text = item.text
        if dec.action == "sanitize" and dec.sanitized_text:
            new_text = dec.sanitized_text
            overrides[item.memory_id] = new_text
        kept_ids.append(item.memory_id)
        kept_items.append(MemoryItem(
            memory_id=item.memory_id,
            text=new_text,
            context=item.context,
            action=item.action,
            outcome=item.outcome,
        ))
    return kept_ids, overrides, kept_items


def apply_write_time_defense(
    case: dict[str, Any],
    defense: MemoryDefense,
    context: DefenseContext,
    log: list[dict[str, Any]],
    *,
    include_controls: bool = True,
    state_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deep copy whose histories are filtered/sanitized. Original case is untouched.

    Each evaluation slot gets `defense.reset_state()` so internal state (lessons,
    caches) is not shared across parallel experimental conditions. Slot names are
    adapter bookkeeping for logging/restore and are never sent to the model.
    """
    adapted = copy.deepcopy(case)
    meta = {
        "family_id": case.get("family_id"),
        "case_id": case.get("case_id"),
        "interface": case.get("interface"),
        "defense": defense.name,
    }
    interface = str(case.get("interface") or "")

    def run_items(items: list[MemoryItem], slot_key: str) -> tuple[list[str], dict[str, str]]:
        defense.reset_state()
        started = time.perf_counter()
        calls_before = defense.llm_calls
        decisions = defense.process_memories(items, context)
        elapsed = time.perf_counter() - started
        actual_calls = defense.llm_calls - calls_before
        apply_amortized_calls(decisions, actual_calls, elapsed)
        if state_sink is not None:
            state_sink[slot_key] = defense.export_state()
        for dec in decisions:
            row = dec.to_log_dict()
            row.update(meta)
            row["condition"] = slot_key
            log.append(row)
        kept_ids, overrides, _ = decisions_to_kept(items, decisions)
        return kept_ids, overrides

    if interface == "P1":
        mb = adapted.setdefault("memory_building", {})
        for condition in ("clean", "poisoned"):
            block = mb.setdefault(f"{condition}_condition", {})
            text = str(block.get("memory_text") or "").strip()
            # Runtime id is anonymous. Condition is only attached to OUTPUT logs.
            item = MemoryItem(memory_id="stored_policy", text=text)
            slot_key = f"learned_{condition}"
            defense.reset_state()
            started = time.perf_counter()
            calls_before = defense.llm_calls
            dec = defense.filter_memory(item, context)
            elapsed = time.perf_counter() - started
            actual_calls = defense.llm_calls - calls_before
            apply_amortized_calls([dec], actual_calls, elapsed)
            if state_sink is not None:
                state_sink[slot_key] = defense.export_state()
            row = dec.to_log_dict()
            row.update(meta)
            row["condition"] = slot_key
            log.append(row)
            if not dec.accepted:
                block["memory_text"] = EMPTY_MEMORY
            elif dec.action == "sanitize" and dec.sanitized_text:
                block["memory_text"] = dec.sanitized_text
        return adapted

    all_overrides: dict[str, str] = {}
    for condition in ("clean", "poisoned"):
        kind, ids = primary_history_ids(adapted, condition)
        items = items_from_ids(adapted, ids)
        kept, overrides = run_items(items, f"learned_{condition}")
        all_overrides.update(overrides)
        _write_history_ids(adapted, condition, kind, kept)

    if include_controls:
        for name, (kind, ids) in control_histories(adapted).items():
            items = items_from_ids(adapted, ids)
            kept, overrides = run_items(items, f"mechanism_control:{name}")
            all_overrides.update(overrides)
            block = adapted.setdefault("memory_building", {}).setdefault("mechanism_controls", {}).setdefault(name, {})
            if kind == "experience_ids":
                block["experience_ids"] = list(kept)
                block["experience_budget"] = len(kept)
            elif kind == "order":
                block["order"] = list(kept)
            elif kind == "selected_ids":
                block["selected_ids"] = list(kept)
                block["order"] = list(kept)

    _apply_text_overrides(adapted, all_overrides)
    return adapted


class DefenseSession:
    """Per-case adapter used by the benchmark runner. Never mutates the frozen case."""

    def __init__(self, defense: MemoryDefense, case: dict[str, Any]):
        self.defense = defense
        self.case = case
        self.log: list[dict[str, Any]] = []
        self._log_start = 0
        self._llm_start = int(getattr(defense, "llm_calls", 0) or 0)
        self._task_cache: dict[str, tuple[str, str, dict[str, Any], dict[str, dict[str, Any]]]] = {}
        self._prepared_case: dict[str, Any] | None = None
        self._slot_states: dict[str, Any] = {}
        self.defense.begin_case(str(case.get("case_id") or ""))

    def attach_shared_log(self, log: list[dict[str, Any]]) -> None:
        self.log = log
        self._log_start = len(log)

    @property
    def per_task(self) -> bool:
        return bool(getattr(self.defense, "per_task", False))

    @property
    def applies_write_time(self) -> bool:
        return bool(getattr(self.defense, "applies_write_time", True))

    @property
    def intercepts_agent_generation(self) -> bool:
        return bool(getattr(self.defense, "intercepts_agent_generation", False))

    def prepare_case(self, task: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.applies_write_time:
            return self.case
        context = defense_context_from_case(self.case, task)
        return apply_write_time_defense(
            self.case, self.defense, context, self.log, state_sink=self._slot_states,
        )

    def case_for_build(self) -> dict[str, Any]:
        if self.per_task or not self.applies_write_time:
            return self.case
        if self._prepared_case is None:
            self._prepared_case = self.prepare_case(task=None)
        return self._prepared_case

    def begin_query(self) -> None:
        """Reset defense state for one downstream query. No evaluation labels."""
        self.defense.reset_state()

    def build_for_task(
        self,
        task: dict[str, Any],
        build_learned_memories: Callable[..., tuple[str, str, dict[str, Any], dict[str, dict[str, Any]]]],
        memory_model: Any,
        max_words: int,
        p5_updater: str,
        p5_random_orders: int,
        random_seed: int,
        build_mechanism_controls: bool,
    ) -> tuple[str, str, dict[str, Any], dict[str, dict[str, Any]]]:
        task_id = str(task.get("task_id") or "")
        if task_id in self._task_cache:
            return self._task_cache[task_id]
        self.defense.reset_state()
        adapted = self.prepare_case(task=task)
        result = build_learned_memories(
            memory_model,
            adapted,
            max_words,
            p5_updater,
            p5_random_orders=p5_random_orders,
            random_seed=random_seed,
            build_mechanism_controls=build_mechanism_controls,
        )
        self._task_cache[task_id] = result
        return result

    def prepare_agent_memory(
        self,
        memory_text: str | None,
        task: dict[str, Any],
        condition: str,
    ) -> str | None:
        context = defense_context_from_case(self.case, task)
        # Adapter-only lookup. `condition` is never sent to a defense prompt.
        if self.applies_write_time:
            state = self._slot_states.get(condition)
            if state is None and condition in {"clean", "poisoned"}:
                state = self._slot_states.get(f"learned_{condition}")
            if state is not None:
                self.defense.import_state(state)
        return self.defense.prepare_agent_memory(memory_text, context)

    def make_generation_interceptor(
        self,
        task: dict[str, Any],
        *,
        original_memory: str | None = None,
        task_format: str = "mcq",
        input_overhead: dict[str, Any] | None = None,
    ) -> Callable[..., str] | None:
        if not self.intercepts_agent_generation:
            return None
        context = defense_context_from_case(self.case, task)
        choice_map = None
        if context.choices:
            choice_map = {
                str(c.get("label") or ""): str(c.get("text") or "")
                for c in context.choices
                if c.get("label")
            }
        runtime = AgentRuntimeInput(
            context=context,
            memory_text=original_memory,
            task_format=task_format,
            choice_map=choice_map,
            tool_calls=[],
        )
        outer_input_overhead = dict(input_overhead or {})

        def _interceptor(generate: Callable[[str, str], str], system: str, user: str) -> str:
            started = time.perf_counter()
            calls_before = int(getattr(self.defense, "llm_calls", 0) or 0)
            raw, diag = self.defense.intercept_agent_generation(
                generate, system, user, runtime,
            )
            elapsed = time.perf_counter() - started
            actual_calls = int(getattr(self.defense, "llm_calls", 0) or 0) - calls_before
            if not isinstance(diag, dict):
                diag = {}
            if not diag.get("input_overhead"):
                diag = {**diag, "input_overhead": outer_input_overhead}
            if not diag.get("output_overhead"):
                diag = {
                    **diag,
                    "output_overhead": {
                        "llm_calls": actual_calls,
                        "latency_seconds": elapsed,
                    },
                }
            _interceptor.last_diag = diag
            self._log_generation_diag(diag, elapsed, actual_calls, original_memory)
            return raw

        _interceptor.last_diag = {}
        return _interceptor

    def _log_generation_diag(
        self,
        diag: dict[str, Any],
        elapsed: float,
        actual_calls: int,
        original_memory: str | None,
    ) -> None:
        meta = {
            "family_id": self.case.get("family_id"),
            "case_id": self.case.get("case_id"),
            "interface": self.case.get("interface"),
            "defense": getattr(self.defense, "name", "unknown"),
            "display_name": getattr(self.defense, "display_name", "") or getattr(self.defense, "name", ""),
        }
        input_diag = diag.get("input_check") or {}
        output_diag = diag.get("output_check") or {}
        user_diag = diag.get("user_task_extraction") or {}
        input_feedback = bool(input_diag.get("feedback"))
        blocked = bool(output_diag.get("blocked"))
        input_dec = DefenseDecision(
            memory_id="external_memory",
            accepted=True,
            action="feedback" if input_feedback else "keep",
            score=None,
            reason="input-side Task Shield check; original external content retained",
            original_text=str(original_memory or ""),
            sanitized_text=None,
            extra={
                "stage": "input",
                "user_tasks": user_diag.get("user_tasks"),
                "user_task_parse_failure": user_diag.get("parse_failure"),
                "empty_user_task_set": user_diag.get("empty_user_task_set"),
                "extracted_instructions": input_diag.get("extracted_instructions"),
                "empty_instruction_set": input_diag.get("empty_instruction_set"),
                "extraction_parse_failure": input_diag.get("extraction_parse_failure"),
                "confirmed_safe": input_diag.get("confirmed_safe"),
                "checks": input_diag.get("checks"),
                "misaligned_instructions": input_diag.get("misaligned_instructions"),
                "feedback": input_diag.get("feedback"),
                "original_retained": True,
                "alignment_rule": diag.get("alignment_rule"),
                "prompt_version": diag.get("prompt_version"),
                "model_pairing": diag.get("model_pairing"),
                "stage_llm_calls": int((diag.get("input_overhead") or {}).get("llm_calls") or 0),
                "stage_latency_seconds": float((diag.get("input_overhead") or {}).get("latency_seconds") or 0.0),
            },
        )
        output_status = str(output_diag.get("final_status") or "aligned")
        output_dec = DefenseDecision(
            memory_id="assistant_output",
            accepted=not blocked,
            action="block" if blocked else "accept",
            score=None,
            reason=f"output-side Task Shield check: {output_status}",
            original_text="",
            extra={
                "stage": "output",
                "user_tasks": user_diag.get("user_tasks"),
                "history": output_diag.get("history"),
                "correction_rounds": output_diag.get("correction_rounds"),
                "max_correction_rounds": output_diag.get("max_correction_rounds"),
                "max_correction_rounds_source": output_diag.get("max_correction_rounds_source"),
                "parse_failures": output_diag.get("parse_failures"),
                "recovered_parse": output_diag.get("recovered_parse"),
                "technical_failure": output_diag.get("technical_failure"),
                "final_status": output_status,
                "outcome_class": output_diag.get("outcome_class"),
                "blocked": blocked,
                "pre_block_raw": output_diag.get("pre_block_raw"),
                "tool_call_branch_applicable": output_diag.get("tool_call_branch_applicable"),
                "tool_call_adaptation": diag.get("tool_call_adaptation"),
                "alignment_rule": diag.get("alignment_rule"),
                "prompt_version": diag.get("prompt_version"),
                "model_pairing": diag.get("model_pairing"),
                "stage_llm_calls": int((diag.get("output_overhead") or {}).get("llm_calls") or 0),
                "stage_latency_seconds": float((diag.get("output_overhead") or {}).get("latency_seconds") or 0.0),
            },
        )
        input_oh = diag.get("input_overhead")
        output_oh = diag.get("output_overhead")
        if input_oh is not None or output_oh is not None:
            apply_stage_overhead(input_dec, input_oh if isinstance(input_oh, dict) else {})
            apply_stage_overhead(output_dec, output_oh if isinstance(output_oh, dict) else {})
        else:
            apply_amortized_calls([input_dec, output_dec], actual_calls, elapsed)
        for dec in (input_dec, output_dec):
            row = dec.to_log_dict()
            row.update(meta)
            self.log.append(row)

    def case_overhead(self) -> dict[str, Any]:
        rows = self.log[self._log_start:]
        stats = summarize_defense_log(rows)
        stats["total_llm_calls"] = int(getattr(self.defense, "llm_calls", 0) or 0) - self._llm_start
        return stats


def prepare_query_execution(
    session: DefenseSession | None,
    memory: str | None,
    task: dict[str, Any],
    condition: str,
    *,
    task_format: str = "mcq",
) -> tuple[str | None, Callable[..., str] | None]:
    """Reset per-query state, optionally check memory, and return a generation interceptor.

    Outer prepare splices input-side feedback into the memory string used by
    `task_prompt`. Input-stage LLM calls (task extraction + memory check) are
    counted here, before the interceptor starts, so decision logs do not
    amortize later output-stage calls onto the input row. The interceptor is
    the failsafe: if prepare did not run, it still applies that same feedback
    to the user payload before generate().
    """
    if session is None:
        return memory, None
    original = memory
    input_overhead = {"llm_calls": 0, "latency_seconds": 0.0}
    if session.intercepts_agent_generation:
        session.begin_query()
    should_prepare = memory is not None and (
        session.intercepts_agent_generation
        or str(condition).startswith("learned_")
        or str(condition).startswith("mechanism_control:")
    )
    mem = memory
    if should_prepare:
        calls_before = int(getattr(session.defense, "llm_calls", 0) or 0)
        started = time.perf_counter()
        mem = session.prepare_agent_memory(memory, task, condition)
        input_overhead = {
            "llm_calls": int(getattr(session.defense, "llm_calls", 0) or 0) - calls_before,
            "latency_seconds": time.perf_counter() - started,
        }
    interceptor = session.make_generation_interceptor(
        task,
        original_memory=original,
        task_format=task_format,
        input_overhead=input_overhead,
    )
    return mem, interceptor


def cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_defense_log(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_kept = n_rejected = n_sanitized = n_quarantined = n_bypassed = 0
    n_feedback = n_blocked = 0
    actions: dict[str, int] = {}
    batch_call_events = 0.0
    # Sum amortized shares; this equals actual calls if accounting is consistent.
    amortized_sum = 0.0
    latency_sum = 0.0
    for row in rows:
        act = str(row.get("action") or row.get("decision") or "")
        actions[act] = actions.get(act, 0) + 1
        if act in {"keep", "accept"}:
            n_kept += 1
        elif act == "reject":
            n_rejected += 1
        elif act == "sanitize":
            n_sanitized += 1
        elif act == "quarantine":
            n_quarantined += 1
        elif act in {"bypass", "parse_failure_bypass"}:
            n_bypassed += 1
        elif act == "feedback":
            n_feedback += 1
        elif act == "block":
            n_blocked += 1
        amortized_sum += float(row.get("amortized_llm_calls") or 0.0)
        latency_sum += float(row.get("latency") or 0.0)
        batch_call_events += float(row.get("batch_llm_calls") or 0.0)

    # batch_llm_calls is duplicated on every memory in a batch; do not sum it.
    # amortized_sum is the correct reconstruction of actual calls from memory rows.
    return {
        "n_decisions": len(rows),
        "n_kept": n_kept,
        "n_rejected": n_rejected,
        "n_sanitized": n_sanitized,
        "n_quarantined": n_quarantined,
        "n_bypassed": n_bypassed,
        "n_feedback": n_feedback,
        "n_blocked": n_blocked,
        "n_modified": n_sanitized,
        "by_action": actions,
        "total_llm_calls": int(round(amortized_sum)),
        "latency_seconds": round(latency_sum, 4),
        "num_memories": len(rows),
    }


def is_defense_blocked_output(raw: str | None) -> bool:
    text = str(raw or "")
    return text.startswith(DEFENSE_BLOCKED_SENTINEL) or DEFENSE_BLOCKED_SENTINEL in text[:160]


def interceptor_output_meta(interceptor: Any) -> dict[str, Any]:
    if interceptor is None:
        return {}
    diag = getattr(interceptor, "last_diag", None)
    if not isinstance(diag, dict):
        return {}
    output = diag.get("output_check") or {}
    outcome = output.get("outcome_class")
    parse_failures = int(output.get("parse_failures") or 0)
    technical = bool(output.get("technical_failure")) or str(outcome or "") in TECHNICAL_FAILURE_CLASSES
    recovered = bool(output.get("recovered_parse")) or (parse_failures > 0 and not technical)
    return {
        "defense_blocked": bool(output.get("blocked")),
        "defense_final_status": output.get("final_status"),
        "defense_outcome_class": outcome,
        "pre_block_raw": output.get("pre_block_raw"),
        "correction_rounds": output.get("correction_rounds"),
        "defense_parse_failures": parse_failures,
        "defense_recovered_parse": recovered,
        "defense_technical_failure": technical,
    }


def defense_result_fields(meta: dict[str, Any] | None) -> dict[str, Any]:
    payload = meta or {}
    return {
        "defense_blocked": bool(payload.get("defense_blocked")),
        "defense_final_status": payload.get("defense_final_status"),
        "defense_outcome_class": payload.get("defense_outcome_class"),
        "pre_block_raw": payload.get("pre_block_raw"),
        "correction_rounds": payload.get("correction_rounds"),
        "defense_parse_failures": int(payload.get("defense_parse_failures") or 0),
        "defense_recovered_parse": bool(payload.get("defense_recovered_parse")),
        "defense_technical_failure": bool(payload.get("defense_technical_failure")),
    }


def apply_blocked_prediction(
    raw: str | None,
    pred: str | None,
    parse_method: str | None,
    interceptor: Any = None,
) -> tuple[str | None, str, dict[str, Any]]:
    """Blocked utterances stay in the denominator and are not parsed as answers."""
    meta = interceptor_output_meta(interceptor)
    blocked = bool(meta.get("defense_blocked")) or is_defense_blocked_output(raw)
    if not blocked:
        return pred, str(parse_method or ""), meta
    meta = {**meta, "defense_blocked": True}
    return None, "defense_blocked", meta
