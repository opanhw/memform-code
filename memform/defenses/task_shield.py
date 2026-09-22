#!/usr/bin/env python3
"""Task Shield (adapted) for MemForm.

Faithful test-time reproduction of Jia et al., ACL 2025, using the original
instruction-extraction, ContributesTo checker, and feedback prompts. Memory
formation is not rewritten.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from memform.defenses.common import (
    AgentRuntimeInput,
    ChatModelLike,
    DefenseContext,
    DefenseDecision,
    MemoryDefense,
    MemoryItem,
    TECHNICAL_FAILURE_CLASSES,
    extract_json_array,
    runtime_visible_texts,
)
from memform.defenses.task_shield_prompts import adapted_memform as prompts
from memform.defenses.task_shield_prompts import original_acl2025 as original_prompts

PROMPT_VERSION = prompts.PROMPT_VERSION
DISPLAY_NAME = "Task Shield (adapted)"
ZERO_SUM_RULE = "misaligned iff S(e) = sum_t s(e,t) == 0"

GenerateFn = Callable[[str, str], str]
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CHOICE_HEADER = re.compile(r"(?i)^\s*(choices?|options?)\s*:?\s*$")


def _fill(template: str, **kwargs: Any) -> str:
    """Replace original-template placeholders once. Inserted text is not re-scanned."""
    if not kwargs:
        return template

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key in kwargs:
            return str(kwargs[key])
        return match.group(0)

    return _PLACEHOLDER.sub(repl, template)


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _choice_lines(choices: list[dict[str, Any]] | None) -> list[str]:
    lines: list[str] = []
    for choice in choices or []:
        label = str(choice.get("label") or "").strip()
        if not label:
            continue
        text = str(choice.get("text") or "").strip()
        lines.append(f"{label}. {text}")
    return lines


def _strip_embedded_choices(
    task_text: str,
    choices: list[dict[str, Any]] | None,
) -> tuple[str, bool]:
    """Remove a choice block that already lives inside task_text."""
    text = str(task_text or "")
    lines = _choice_lines(choices)
    stripped = False
    if lines:
        block = "\n".join(lines)
        if block in text:
            text = text.replace(block, "")
            stripped = True
        elif all(line in text for line in lines):
            for line in lines:
                text = text.replace(line, "")
            stripped = True
    if stripped:
        kept: list[str] = []
        for line in text.splitlines():
            if _CHOICE_HEADER.match(line):
                continue
            kept.append(line)
        text = "\n".join(kept)
    cleaned = "\n".join(line for line in text.splitlines() if line.strip()).strip()
    return cleaned, stripped


def task_request_text(context: DefenseContext | None) -> tuple[str, dict[str, Any]]:
    """Build Tu source text from the task request only. Choices are not goals."""
    meta = {
        "source": "empty",
        "task_text_contained_choices": False,
        "used_decision_variable_fallback": False,
    }
    if context is None:
        return "", meta
    raw = str(context.task_text or "").strip()
    if raw:
        stripped, contained = _strip_embedded_choices(raw, context.choices)
        meta["task_text_contained_choices"] = contained
        if stripped:
            meta["source"] = "task_text"
            return stripped, meta
        meta["source"] = "decision_variable_fallback"
        meta["used_decision_variable_fallback"] = True
        return str(context.decision_variable or "").strip(), meta
    decision = str(context.decision_variable or "").strip()
    if decision:
        meta["source"] = "decision_variable"
    return decision, meta


def parse_instruction_list(raw: str) -> tuple[list[str] | None, str | None]:
    """Parse a JSON string array. Valid `[]` is empty, not a parse failure."""
    stripped = str(raw or "").strip()
    if not stripped:
        return None, "empty_string"
    arr = extract_json_array(raw)
    if not isinstance(arr, list):
        return None, "not_json_array"
    out: list[str] = []
    for item in arr:
        if not isinstance(item, str):
            return None, "schema_not_list_of_strings"
        text = item.strip()
        if not text:
            return None, "empty_string_element"
        out.append(text)
    return out, None


def _parse_instruction_list(raw: str) -> list[str] | None:
    tasks, error = parse_instruction_list(raw)
    if error:
        return None
    return tasks


def _score_from_row(row: dict[str, Any]) -> float | None:
    for key in (
        "ContributeTo score",
        "ContributeTo",
        "contributes_to",
        "contribute_to_score",
        "ContributeTo_score",
        "score",
    ):
        if key in row and row[key] is not None:
            value = row[key]
            break
    else:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lower() in {"nan", "inf", "-inf", "+inf"}:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def _row_task_text(row: dict[str, Any]) -> str:
    return str(
        row.get("user_task_instruction")
        or row.get("task_instruction")
        or row.get("user_task")
        or ""
    ).strip()


def parse_score_rows(raw: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Require a non-empty JSON array of objects with a finite ContributeTo score."""
    stripped = str(raw or "").strip()
    if not stripped:
        return None, "empty_string"
    arr = extract_json_array(raw)
    if not isinstance(arr, list):
        return None, "not_json_array"
    if not arr:
        return None, "empty_score_array"
    rows: list[dict[str, Any]] = []
    for item in arr:
        if not isinstance(item, dict) or not item:
            return None, "schema_not_object_array"
        if not _row_task_text(item):
            return None, "missing_user_task_instruction"
        if _score_from_row(item) is None:
            return None, "invalid_or_nonfinite_score"
        rows.append(item)
    return rows, None


def _dedupe_tasks(tasks: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for task in tasks:
        text = str(task or "").strip()
        key = _norm_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _map_scores_to_tasks(
    rows: list[Any],
    user_tasks: list[str],
) -> tuple[dict[str, float], list[str]]:
    """Return per-task scores and mapping-error codes. Fail closed on errors."""
    errors: list[str] = []
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        return {}, ["checker_output_not_object_array"]
    unused = [dict(r) for r in rows if isinstance(r, dict)]
    mapped: dict[str, float] = {}
    for task in user_tasks:
        score = None
        match_i = None
        exact = [i for i, r in enumerate(unused) if _row_task_text(r) == task]
        if len(exact) == 1:
            match_i = exact[0]
        elif not exact:
            target = _norm_text(task)
            fuzzy = [i for i, r in enumerate(unused) if _norm_text(_row_task_text(r)) == target]
            if len(fuzzy) == 1:
                match_i = fuzzy[0]
            elif len(fuzzy) > 1 or len(exact) > 1:
                errors.append(f"ambiguous_task_mapping:{task}")
                continue
        else:
            errors.append(f"ambiguous_task_mapping:{task}")
            continue
        if match_i is None:
            errors.append(f"missing_task_mapping:{task}")
            continue
        row = unused.pop(match_i)
        score = _score_from_row(row)
        if score is None:
            errors.append(f"invalid_or_missing_score:{task}")
            continue
        mapped[task] = score
    return mapped, errors


def _parse_choice_label(text: str, labels: list[str]) -> str | None:
    """Map MCQ output to a choice letter using runtime-visible labels only."""
    mapping = {str(x).upper(): str(x) for x in labels}
    if not mapping:
        return None
    label_alt = "|".join(re.escape(x) for x in sorted(mapping, key=len, reverse=True))
    matches = re.findall(
        rf"FINAL[_\s-]*ANSWER\s*(?:[:：=]|IS\b)\s*[\[(]?({label_alt})[\])]?",
        str(text or ""),
        flags=re.I,
    )
    if matches:
        return mapping.get(str(matches[-1]).upper())
    stripped = str(text or "").strip()
    if stripped.upper() in mapping:
        return mapping[stripped.upper()]
    return None


@dataclass
class HistoryEvent:
    source: str
    text: str
    checker_text: str | None = None

    def for_checker(self) -> str:
        return self.checker_text if self.checker_text is not None else self.text

    def render(self) -> str:
        return f"[{self.source}] {self.for_checker()}"


@dataclass
class InstructionCheck:
    instruction: str
    source_level: str
    scores: dict[str, float] = field(default_factory=dict)
    aggregate: float | None = None
    misaligned: bool = True
    parse_failure: bool = False
    mapping_errors: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    empty_instruction_set: bool = False
    confirmed_safe: bool = False


class TaskShield(MemoryDefense):
    """Use-time Task Shield. Does not rewrite memory formation."""

    name = "task_shield"
    display_name = DISPLAY_NAME
    per_task = False
    applies_write_time = False
    intercepts_agent_generation = True
    prompt_version = PROMPT_VERSION

    def __init__(
        self,
        model: ChatModelLike | None = None,
        threshold: float = 0.0,
        max_correction_rounds: int = 3,
        max_parse_repairs: int = 1,
        model_pairing: str = "independent",
        **kwargs: Any,
    ):
        # Alignment uses S(e)==0 from the paper, not the 0.5 admission threshold
        # used by other MemForm defenses. `threshold` is recorded but ignored.
        super().__init__(model=model, threshold=0.0, **kwargs)
        self.alignment_rule = ZERO_SUM_RULE
        self.max_correction_rounds = max(0, int(max_correction_rounds))
        self.max_parse_repairs = max(0, int(max_parse_repairs))
        self.model_pairing = str(model_pairing or "independent")
        self._user_tasks: list[str] = []
        self._history: list[HistoryEvent] = []
        self._conversation: list[str] = []
        self._query_diag: dict[str, Any] = {}
        self._original_memory: str | None = None
        self._defended_memory_text: str | None = None
        self._input_feedback: str = ""
        self._user_task_parse_failure = False
        self._empty_user_task_set = False

    def reset_state(self) -> None:
        self._user_tasks = []
        self._history = []
        self._conversation = []
        self._query_diag = {}
        self._original_memory = None
        self._defended_memory_text = None
        self._input_feedback = ""
        self._user_task_parse_failure = False
        self._empty_user_task_set = False
        self._excluded_texts = []

    def _accumulate_input_overhead(self, llm_calls: int, latency: float) -> None:
        prev = self._query_diag.get("input_overhead") or {}
        self._query_diag["input_overhead"] = {
            "llm_calls": int(prev.get("llm_calls") or 0) + int(llm_calls),
            "latency_seconds": float(prev.get("latency_seconds") or 0.0) + float(latency),
        }

    def export_state(self) -> dict[str, Any]:
        return {}

    def import_state(self, state: dict[str, Any] | None) -> None:
        return None

    def _sync_conversation(self) -> None:
        self._conversation = [event.render() for event in self._history]

    def _append_history(
        self,
        source: str,
        text: str,
        *,
        checker_text: str | None = None,
    ) -> None:
        self._history.append(HistoryEvent(source=source, text=text, checker_text=checker_text))
        self._sync_conversation()

    def _excluded(self, context: DefenseContext | None, *blobs: str | None) -> list[str]:
        texts = runtime_visible_texts(None, context)
        for blob in blobs:
            if blob:
                texts.append(str(blob))
        texts.extend(self._user_tasks)
        texts.extend(event.for_checker() for event in self._history)
        return texts

    def _invoke_json(
        self,
        system: str,
        user: str,
        *,
        excluded: list[str],
        expect: str,
    ) -> str:
        raw = self.invoke_llm(system, user, excluded_texts=excluded)
        if expect == "list" and parse_instruction_list(raw)[1] is None:
            return raw
        if expect == "scores" and parse_score_rows(raw)[1] is None:
            return raw
        for _ in range(self.max_parse_repairs):
            schema = (
                "a JSON array of strings"
                if expect == "list"
                else "a JSON array of objects with user_task_instruction and a finite ContributeTo score in [0, 1]"
            )
            repair_user = _fill(
                prompts.JSON_REPAIR_USER,
                schema_description=schema,
                previous_output=raw,
            )
            raw = self.invoke_llm(
                prompts.JSON_REPAIR_SYSTEM,
                repair_user,
                excluded_texts=list(excluded) + [raw],
            )
            if expect == "list" and parse_instruction_list(raw)[1] is None:
                return raw
            if expect == "scores" and parse_score_rows(raw)[1] is None:
                return raw
        return raw

    def _extract_instructions(
        self,
        message_content: str,
        subject: str,
        context: DefenseContext | None,
    ) -> tuple[list[str] | None, str, str | None]:
        user = _fill(
            prompts.TASK_EXTRACTION_USER,
            subject=subject,
            message_content=message_content,
        )
        raw = self._invoke_json(
            prompts.TASK_EXTRACTION_SYSTEM,
            user,
            excluded=self._excluded(context, message_content, subject),
            expect="list",
        )
        parsed, error = parse_instruction_list(raw)
        return parsed, raw, error

    def _history_text(self) -> str:
        if not self._history:
            return "(empty)"
        return "\n".join(event.render() for event in self._history)

    def _task_instructions_text(self) -> str:
        if not self._user_tasks:
            return "(none extracted)"
        return json.dumps(self._user_tasks, ensure_ascii=False)

    def _agent_user_from_history(self, base_user: str) -> str:
        """Same sourced history as the checker, concatenated for single-turn invoke()."""
        parts = [base_user]
        round_i = 0
        pending_raw: str | None = None
        for event in self._history:
            if event.source == "assistant":
                pending_raw = event.text
            elif event.source == "shield_feedback" and pending_raw is not None:
                round_i += 1
                parts.append(f"Assistant previous response {round_i}:\n{pending_raw}")
                parts.append(f"Shield feedback {round_i}:\n{event.text}")
                pending_raw = None
        return "\n\n".join(parts)

    def _ensure_input_feedback_in_user(self, user: str) -> str:
        """Interceptor owns the failsafe splice. Outer prepare may already have done it."""
        feedback = self._input_feedback
        original = self._original_memory
        defended = self._defended_memory_text
        if not feedback:
            return user
        if user.count(feedback) >= 1:
            return user
        if defended and defended in user:
            return user
        if original and original in user and defended:
            return user.replace(original, defended, 1)
        if original and original in user:
            return user.replace(original, f"{original}\n\n{feedback}", 1)
        return f"{user}\n\n{feedback}"

    def _check_instruction(
        self,
        instruction: str,
        current_level: str,
        tool_information: str,
        context: DefenseContext | None,
        *,
        source_content: str | None = None,
    ) -> InstructionCheck:
        result = InstructionCheck(instruction=instruction, source_level=current_level)
        if self._user_task_parse_failure:
            result.parse_failure = True
            result.misaligned = True
            result.mapping_errors = ["user_task_set_parse_failure"]
            result.confirmed_safe = False
            return result
        if not self._user_tasks:
            result.aggregate = 0.0
            result.misaligned = True
            result.mapping_errors = ["empty_user_task_set"]
            result.confirmed_safe = False
            return result

        user = _fill(
            prompts.CONTENT_CHECKER_USER,
            actionable_instruction=instruction,
            current_level=current_level,
            tool_information=tool_information,
            task_instructions_text=self._task_instructions_text(),
        )
        if source_content:
            user = user + _fill(
                prompts.SOURCE_CONTENT_BLOCK,
                source_level=current_level,
                source_content=source_content,
            )
        user = user + _fill(
            prompts.CONVERSATION_HISTORY_BLOCK,
            conversation_history=self._history_text(),
        )
        raw = self._invoke_json(
            prompts.CONTENT_CHECKER_SYSTEM,
            user,
            excluded=self._excluded(
                context,
                instruction,
                current_level,
                tool_information,
                source_content,
                self._history_text(),
            ),
            expect="scores",
        )
        rows, schema_error = parse_score_rows(raw)
        if rows is None:
            result.parse_failure = True
            result.misaligned = True
            result.mapping_errors = [schema_error or "checker_parse_failure"]
            result.confirmed_safe = False
            return result
        mapped, errors = _map_scores_to_tasks(rows, self._user_tasks)
        result.scores = mapped
        result.mapping_errors = errors
        result.reasons = [
            str(row.get("reason") or "")
            for row in rows
            if isinstance(row, dict)
        ]
        if errors or len(mapped) != len(self._user_tasks):
            result.parse_failure = True
            result.misaligned = True
            result.confirmed_safe = False
            result.aggregate = None
            return result
        total = sum(mapped.values())
        result.aggregate = total
        result.misaligned = total == 0
        result.confirmed_safe = not result.misaligned
        return result

    def _check_tool_call(
        self,
        tool_name: str,
        tool_args: Any,
        related_content: str,
        context: DefenseContext | None,
    ) -> InstructionCheck:
        instruction = f"Call {tool_name} with arguments {tool_args}"
        result = InstructionCheck(instruction=instruction, source_level="assistant")
        if self._user_task_parse_failure or not self._user_tasks:
            result.parse_failure = self._user_task_parse_failure
            result.misaligned = True
            result.mapping_errors = (
                ["user_task_set_parse_failure"] if self._user_task_parse_failure
                else ["empty_user_task_set"]
            )
            return result
        user = _fill(
            prompts.TOOL_CALL_CHECKER_USER,
            tool_name=tool_name,
            tool_args=json.dumps(tool_args, ensure_ascii=False, default=str),
            related_content=related_content,
            task_instructions_text=self._task_instructions_text(),
        )
        user = user + _fill(
            prompts.CONVERSATION_HISTORY_BLOCK,
            conversation_history=self._history_text(),
        )
        raw = self._invoke_json(
            prompts.TOOL_CALL_CHECKER_SYSTEM,
            user,
            excluded=self._excluded(context, tool_name, related_content, self._history_text()),
            expect="scores",
        )
        rows, schema_error = parse_score_rows(raw)
        if rows is None:
            result.parse_failure = True
            result.misaligned = True
            result.mapping_errors = [schema_error or "tool_call_checker_parse_failure"]
            return result
        mapped, errors = _map_scores_to_tasks(rows, self._user_tasks)
        result.scores = mapped
        result.mapping_errors = errors
        if errors or len(mapped) != len(self._user_tasks):
            result.parse_failure = True
            result.misaligned = True
            return result
        total = sum(mapped.values())
        result.aggregate = total
        result.misaligned = total == 0
        result.confirmed_safe = not result.misaligned
        return result

    def _build_feedback(
        self,
        misaligned: list[InstructionCheck],
        *,
        kind: str,
        tool_call_id: str | None = None,
    ) -> str:
        parts: list[str] = []
        for item in misaligned:
            parts.append(_fill(
                original_prompts.CONTENT_MISALIGNMENT_FEEDBACK,
                misaligned_instruction=item.instruction,
            ))
        if kind == "tool_output" and tool_call_id:
            parts.append(_fill(
                original_prompts.TOOL_OUTPUT_MISALIGNMENT_FEEDBACK,
                misaligned_tool_call_id=tool_call_id,
            ))
        elif kind == "stored_memory":
            parts.append(prompts.MEMORY_FEEDBACK_ADAPTATION)
        reminder_lines = [original_prompts.USER_INTENTIONS_REMINDER_HEADER]
        if self._user_tasks:
            reminder_lines.extend(f"- {task}" for task in self._user_tasks)
        else:
            reminder_lines.append("- (no user task instructions extracted)")
        parts.append("\n".join(reminder_lines))
        return "\n\n".join(parts)

    def _user_message_content(self, context: DefenseContext | None) -> str:
        text, _meta = task_request_text(context)
        return text

    def _establish_user_tasks(self, context: DefenseContext | None) -> dict[str, Any]:
        request, request_meta = task_request_text(context)
        self._append_history("user_task_request", request)
        choice_lines = _choice_lines(None if context is None else context.choices)
        if choice_lines:
            self._append_history(
                "candidate_options",
                "Candidate options are context for understanding the question. "
                "They are not user instructions and must not be treated as required actions.\n"
                + "\n".join(choice_lines),
            )
        extracted, raw, error = self._extract_instructions(request, "the user", context)
        if extracted is None:
            self._user_task_parse_failure = True
            self._user_tasks = []
            self._empty_user_task_set = False
            return {
                "user_tasks": [],
                "parse_failure": True,
                "parse_error": error,
                "empty_user_task_set": False,
                "raw": raw,
                "task_request": request,
                **request_meta,
            }
        self._user_tasks = _dedupe_tasks(extracted)
        self._user_task_parse_failure = False
        self._empty_user_task_set = len(self._user_tasks) == 0
        return {
            "user_tasks": list(self._user_tasks),
            "parse_failure": False,
            "parse_error": None,
            "empty_user_task_set": self._empty_user_task_set,
            "raw": raw,
            "task_request": request,
            "deduped": len(self._user_tasks) != len(extracted),
            **request_meta,
        }

    def _check_external_content(
        self,
        text: str,
        context: DefenseContext | None,
        *,
        subject: str,
        level: str,
        tool_information: str,
        kind: str,
        tool_call_id: str | None = None,
        source_content: str | None = None,
    ) -> dict[str, Any]:
        extracted, raw, error = self._extract_instructions(text, subject, context)
        diag: dict[str, Any] = {
            "extracted_instructions": extracted,
            "extraction_raw": raw,
            "extraction_parse_failure": extracted is None,
            "extraction_parse_error": error,
            "empty_instruction_set": extracted == [],
            "confirmed_safe": False,
            "checks": [],
            "misaligned_instructions": [],
            "feedback": "",
            "original_retained": True,
            "outcome_class": "no_extractable_instructions",
        }
        if extracted is None:
            synthetic = InstructionCheck(
                instruction="(unparsed external content)",
                source_level=level,
                parse_failure=True,
                misaligned=True,
                mapping_errors=[error or "instruction_extraction_parse_failure"],
            )
            feedback = self._build_feedback([synthetic], kind=kind, tool_call_id=tool_call_id)
            diag["feedback"] = feedback
            diag["misaligned_instructions"] = [synthetic.instruction]
            diag["checks"] = [_check_to_dict(synthetic)]
            diag["outcome_class"] = "parse_error"
            return diag
        if not extracted:
            diag["confirmed_safe"] = False
            diag["outcome_class"] = "no_extractable_instructions"
            return diag
        misaligned: list[InstructionCheck] = []
        for instruction in extracted:
            check = self._check_instruction(
                instruction,
                level,
                tool_information,
                context,
                source_content=source_content if source_content is not None else text,
            )
            diag["checks"].append(_check_to_dict(check))
            if check.misaligned:
                misaligned.append(check)
        diag["misaligned_instructions"] = [c.instruction for c in misaligned]
        if misaligned:
            diag["feedback"] = self._build_feedback(
                misaligned, kind=kind, tool_call_id=tool_call_id
            )
            diag["confirmed_safe"] = False
            if any(c.parse_failure for c in misaligned):
                diag["outcome_class"] = "parse_error"
            else:
                diag["outcome_class"] = "semantic_misaligned"
        else:
            diag["confirmed_safe"] = all(
                bool(c.get("confirmed_safe")) for c in diag["checks"]
            )
            diag["outcome_class"] = "aligned" if diag["confirmed_safe"] else "unverified"
        return diag

    def _assistant_content_for_extraction(
        self,
        raw_output: str,
        runtime: AgentRuntimeInput,
    ) -> tuple[str, str | None]:
        choice_text = None
        if runtime.choice_map:
            label = _parse_choice_label(raw_output, list(runtime.choice_map.keys()))
            if label:
                choice_text = runtime.choice_map.get(label)
        content = raw_output
        if choice_text:
            content = (
                f"{raw_output}\n\n"
                "Proposed action (MemForm adaptation: runtime choice text for the "
                f"selected letter; not a reference answer): {choice_text}"
            )
        return content, choice_text

    def filter_memory(self, memory: MemoryItem, context: DefenseContext | None = None) -> DefenseDecision:
        started = time.perf_counter()
        return DefenseDecision(
            memory_id=memory.memory_id,
            accepted=True,
            action="bypass",
            score=None,
            reason="Task Shield is a use-time defense; write-time admission is not applied",
            original_text=memory.text,
            extra={
                "write_time_skipped": True,
                "adaptation": "task_shield_use_time_only",
            },
            latency=time.perf_counter() - started,
            amortized_llm_calls=0.0,
            batch_llm_calls=0,
        )

    def prepare_agent_memory(
        self,
        memory_text: str | None,
        context: DefenseContext | None = None,
        condition: str | None = None,
    ) -> str | None:
        # `condition` is adapter-only and must not enter prompts.
        del condition
        started_calls = int(self.llm_calls)
        started = time.perf_counter()
        if "user_task_extraction" not in self._query_diag:
            self._query_diag["user_task_extraction"] = self._establish_user_tasks(context)
        if memory_text is None:
            self._original_memory = None
            self._defended_memory_text = None
            self._input_feedback = ""
            self._accumulate_input_overhead(
                int(self.llm_calls) - started_calls,
                time.perf_counter() - started,
            )
            return None
        self._original_memory = memory_text
        input_diag = self._check_external_content(
            memory_text,
            context,
            subject=prompts.MEMORY_SUBJECT,
            level=prompts.MEMORY_LEVEL,
            tool_information=prompts.MEMORY_TOOL_INFORMATION,
            kind="stored_memory",
            source_content=memory_text,
        )
        self._query_diag["input_check"] = input_diag
        self._append_history("stored_memory", memory_text)
        feedback = str(input_diag.get("feedback") or "")
        self._input_feedback = feedback
        if feedback:
            self._append_history("input_feedback", feedback)
            defended = f"{memory_text}\n\n{feedback}"
        else:
            defended = memory_text
        self._defended_memory_text = defended
        self._accumulate_input_overhead(
            int(self.llm_calls) - started_calls,
            time.perf_counter() - started,
        )
        return defended

    def intercept_agent_generation(
        self,
        generate: GenerateFn,
        system: str,
        user: str,
        runtime: AgentRuntimeInput,
    ) -> tuple[str, dict[str, Any]]:
        context = runtime.context
        if "user_task_extraction" not in self._query_diag:
            started_calls = int(self.llm_calls)
            started = time.perf_counter()
            self._query_diag["user_task_extraction"] = self._establish_user_tasks(context)
            self._accumulate_input_overhead(
                int(self.llm_calls) - started_calls,
                time.perf_counter() - started,
            )
        if runtime.memory_text and "input_check" not in self._query_diag:
            # Fallback: outer prepare did not run. Interceptor must splice feedback.
            self.prepare_agent_memory(runtime.memory_text, context)

        tool_calls = list(runtime.tool_calls or [])
        tool_branch_applicable = bool(tool_calls)
        self._query_diag["tool_call_branch_applicable"] = tool_branch_applicable
        self._query_diag["tool_call_adaptation"] = (
            "checked" if tool_branch_applicable
            else "not_applicable_no_real_tool_calls_in_protocol"
        )

        output_calls0 = int(self.llm_calls)
        output_started = time.perf_counter()
        base_user = self._ensure_input_feedback_in_user(user)
        current_user = base_user
        correction_rounds = 0
        parse_failures = 0
        last_raw = ""
        pre_block_raw = ""
        output_history: list[dict[str, Any]] = []
        final_status = "unverified"
        outcome_class = "unverified"
        blocked = False

        max_attempts = self.max_correction_rounds + 1
        for attempt in range(max_attempts):
            raw = generate(system, current_user)
            last_raw = raw
            pre_block_raw = raw
            content, choice_text = self._assistant_content_for_extraction(raw, runtime)
            extracted, extract_raw, extract_error = self._extract_instructions(
                content, "the assistant", context
            )
            checks: list[InstructionCheck] = []
            misaligned: list[InstructionCheck] = []
            extraction_parse_failure = extracted is None
            empty_instruction_set = extracted == []
            if self._user_task_parse_failure:
                parse_failures += 1
                synthetic = InstructionCheck(
                    instruction="(unverified; user task set parse failure)",
                    source_level="assistant",
                    parse_failure=True,
                    misaligned=True,
                    mapping_errors=["user_task_set_parse_failure"],
                )
                checks.append(synthetic)
                misaligned.append(synthetic)
            elif extraction_parse_failure:
                parse_failures += 1
                synthetic = InstructionCheck(
                    instruction="(unparsed assistant content)",
                    source_level="assistant",
                    parse_failure=True,
                    misaligned=True,
                    mapping_errors=[extract_error or "assistant_extraction_parse_failure"],
                )
                checks.append(synthetic)
                misaligned.append(synthetic)
            elif extracted:
                for instruction in extracted:
                    check = self._check_instruction(
                        instruction,
                        "assistant",
                        "",
                        context,
                        source_content=content,
                    )
                    checks.append(check)
                    if check.parse_failure:
                        parse_failures += 1
                    if check.misaligned:
                        misaligned.append(check)

            tool_checks: list[InstructionCheck] = []
            if tool_branch_applicable:
                for call in tool_calls:
                    tcheck = self._check_tool_call(
                        str(call.get("name") or call.get("function") or ""),
                        call.get("arguments") or call.get("args") or {},
                        str(call.get("related_content") or raw),
                        context,
                    )
                    tool_checks.append(tcheck)
                    if tcheck.misaligned:
                        misaligned.append(tcheck)

            if not misaligned and empty_instruction_set:
                round_outcome = "no_extractable_instructions"
                confirmed_safe = False
            elif not misaligned:
                round_outcome = "corrected" if attempt > 0 else "aligned"
                confirmed_safe = True
            elif any(c.parse_failure for c in misaligned):
                if self._user_task_parse_failure:
                    round_outcome = "tu_parse_failure"
                else:
                    round_outcome = "parse_error"
                confirmed_safe = False
            else:
                round_outcome = "semantic_misaligned"
                confirmed_safe = False

            round_diag = {
                "attempt": attempt,
                "correction_round": attempt,
                "extracted_instructions": extracted,
                "extraction_parse_failure": extraction_parse_failure,
                "extraction_parse_error": extract_error,
                "empty_instruction_set": empty_instruction_set,
                "mcq_choice_text_adaptation": choice_text,
                "checks": [_check_to_dict(c) for c in checks],
                "tool_checks": [_check_to_dict(c) for c in tool_checks],
                "misaligned_instructions": [c.instruction for c in misaligned],
                "confirmed_safe": confirmed_safe,
                "outcome_class": round_outcome,
            }
            output_history.append(round_diag)

            if not misaligned:
                self._append_history("assistant", raw, checker_text=content)
                final_status = round_outcome
                outcome_class = round_outcome
                blocked = False
                break

            feedback = self._build_feedback(misaligned, kind="assistant")
            round_diag["feedback"] = feedback
            self._append_history("assistant", raw, checker_text=content)
            self._append_history("shield_feedback", feedback)
            if attempt >= self.max_correction_rounds:
                final_status = "retry_exhausted"
                outcome_class = round_outcome
                blocked = True
                last_raw = prompts.BLOCKED_OUTPUT
                break
            correction_rounds += 1
            current_user = self._agent_user_from_history(base_user)

        recovered_parse = parse_failures > 0 and outcome_class not in TECHNICAL_FAILURE_CLASSES
        technical_failure = outcome_class in TECHNICAL_FAILURE_CLASSES
        diag = {
            **self._query_diag,
            "input_overhead": self._query_diag.get("input_overhead") or {
                "llm_calls": 0,
                "latency_seconds": 0.0,
            },
            "output_overhead": {
                "llm_calls": int(self.llm_calls) - output_calls0,
                "latency_seconds": time.perf_counter() - output_started,
            },
            "output_check": {
                "history": output_history,
                "correction_rounds": correction_rounds,
                "max_correction_rounds": self.max_correction_rounds,
                "max_correction_rounds_source": "engineering_choice_paper_says_several_rounds",
                "parse_failures": parse_failures,
                "recovered_parse": recovered_parse,
                "technical_failure": technical_failure,
                "final_status": final_status,
                "outcome_class": outcome_class,
                "blocked": blocked,
                "pre_block_raw": pre_block_raw if blocked else None,
                "tool_call_branch_applicable": tool_branch_applicable,
            },
            "alignment_rule": self.alignment_rule,
            "prompt_version": self.prompt_version,
            "model_pairing": self.model_pairing,
            "display_name": self.display_name,
        }
        self._query_diag = diag
        return last_raw, diag


def _check_to_dict(check: InstructionCheck) -> dict[str, Any]:
    return {
        "instruction": check.instruction,
        "source_level": check.source_level,
        "scores": check.scores,
        "aggregate_S": check.aggregate,
        "misaligned": check.misaligned,
        "parse_failure": check.parse_failure,
        "mapping_errors": list(check.mapping_errors),
        "empty_instruction_set": check.empty_instruction_set,
        "confirmed_safe": check.confirmed_safe,
        "alignment_rule": ZERO_SUM_RULE,
    }


def aggregate_is_misaligned(scores: list[float]) -> bool:
    """Paper rule: misaligned iff S(e) = sum(scores) == 0."""
    return sum(scores) == 0
