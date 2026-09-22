#!/usr/bin/env python3
"""Shared I/O helpers and schema validation for the MemForm benchmark.

Family specs and compiled cases are validated here without calling an LLM.
The compiler, free-form builder, runners, and item auditors all depend on
these checks so causal construction stays model-independent.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_INTERFACES = ("P1", "P2", "P3", "P4", "P5")
ALIGNMENT_LABELS = {"clean", "poisoned", "mixed", "unrelated"}


def repo_root() -> Path:
    """Return the repository root (parent of the ``memform`` package)."""
    return Path(__file__).resolve().parent.parent


def load_env(*, override: bool = True) -> Path:
    """Load ``.env`` from the repository root. Safe to call more than once.

    Also copies ``OPENAI_BASE_URL`` to ``OPENAI_API_BASE`` for LlamaIndex clients.
    """
    root = repo_root()
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(root / ".env", override=override)
    if os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = os.environ["OPENAI_BASE_URL"]
    return root


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: each JSONL row must be an object")
            rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def ensure_choice_label(value: Any, choices: list[dict[str, Any]]) -> str | None:
    norm = normalize_text(value)
    for item in choices:
        label = str(item.get("label", "")).strip()
        text = str(item.get("text", "")).strip()
        if norm in {normalize_text(label), normalize_text(text)}:
            return label
    return None


def _validate_choices(choices: Any, errors: list[str]) -> None:
    if not isinstance(choices, list) or not (2 <= len(choices) <= 5):
        errors.append("choices must contain 2-5 items")
        return
    labels = [str(x.get("label", "")).strip() for x in choices]
    texts = [normalize_text(x.get("text", "")) for x in choices]
    if any(not x for x in labels):
        errors.append("choice labels must be non-empty")
    if len({x.lower() for x in labels}) != len(labels):
        errors.append("choice labels must be unique")
    if any(not x for x in texts):
        errors.append("choice texts must be non-empty")
    if len(set(texts)) != len(texts):
        errors.append("choice texts must be unique")


def _validate_policy(name: str, policy: Any, choices: list[dict[str, Any]], errors: list[str]) -> None:
    if not isinstance(policy, dict):
        errors.append(f"{name} must be an object")
        return
    if not str(policy.get("rule", "")).strip():
        errors.append(f"{name}.rule is required")
    if ensure_choice_label(policy.get("action_label", ""), choices) is None:
        errors.append(f"{name}.action_label must match a choice")


def _validate_trigger(name: str, task: Any, choices: list[dict[str, Any]], errors: list[str]) -> None:
    if not isinstance(task, dict):
        errors.append(f"triggers.{name} is required")
        return
    if not str(task.get("task", "")).strip():
        errors.append(f"triggers.{name}.task is required")
    for field in ("clean_answer", "poisoned_answer"):
        if ensure_choice_label(task.get(field, ""), choices) is None:
            errors.append(f"triggers.{name}.{field} must match a choice")


def validate_family_spec(spec: dict[str, Any]) -> list[str]:
    """Validate a family specification.

    P1-P4 reuse the frozen family semantics in spec.experiences/policies.
    P5 uses spec.p5.experiences and must be same-context conflicting evidence.
    P1/P2 are compiled deterministically and require no additional family fields.
    """
    errors: list[str] = []
    required = [
        "family_id", "domain", "decision_variable", "clean_policy",
        "poisoned_policy", "scope", "choices", "triggers", "experiences", "p5",
    ]
    for key in required:
        if key not in spec:
            errors.append(f"missing field: {key}")
    if errors:
        return errors

    choices = spec["choices"]
    _validate_choices(choices, errors)
    if errors and not isinstance(choices, list):
        return errors

    _validate_policy("clean_policy", spec.get("clean_policy"), choices, errors)
    _validate_policy("poisoned_policy", spec.get("poisoned_policy"), choices, errors)

    scope = spec.get("scope", {})
    for key in ("inside", "near_outside", "target_outside"):
        if not str(scope.get(key, "")).strip():
            errors.append(f"scope.{key} is required")

    triggers = spec.get("triggers", {})
    for name in ("in_scope", "near_out_of_scope", "target_out_of_scope", "general_target"):
        _validate_trigger(name, triggers.get(name), choices, errors)

    ins = triggers.get("in_scope", {})
    if isinstance(ins, dict):
        ca = ensure_choice_label(ins.get("clean_answer", ""), choices)
        pa = ensure_choice_label(ins.get("poisoned_answer", ""), choices)
        if ca and pa and ca != pa:
            errors.append("P3 in_scope clean_answer and poisoned_answer must be identical")
    for name in ("near_out_of_scope", "target_out_of_scope", "general_target"):
        t = triggers.get(name, {})
        if isinstance(t, dict):
            ca = ensure_choice_label(t.get("clean_answer", ""), choices)
            pa = ensure_choice_label(t.get("poisoned_answer", ""), choices)
            if ca and pa and ca == pa:
                errors.append(f"{name} clean_answer and poisoned_answer must differ")

    # P3/P4 evidence pool.
    exps = spec.get("experiences")
    if not isinstance(exps, list) or not exps:
        errors.append("experiences must be a non-empty list")
    else:
        ids: list[str] = []
        support_counts = {"clean": 0, "poisoned": 0, "both": 0, "neutral": 0}
        inside_both = 0
        outside_clean = 0
        for i, exp in enumerate(exps):
            eid = str(exp.get("id", "")).strip()
            if not eid:
                errors.append(f"experiences[{i}].id is required")
            ids.append(eid)
            for field in ("context", "action", "outcome"):
                if not str(exp.get(field, "")).strip():
                    errors.append(f"experience {eid or i}: {field} is required")
            support = str(exp.get("supports", "")).strip().lower()
            if support not in support_counts:
                errors.append(f"experience {eid or i}: supports must be clean/poisoned/both/neutral")
            else:
                support_counts[support] += 1
            bucket = str(exp.get("scope_bucket", "")).strip()
            if bucket not in {"inside", "near_outside", "target_outside", "general"}:
                errors.append(f"experience {eid or i}: invalid scope_bucket={bucket!r}")
            if bucket == "inside" and support == "both" and bool(exp.get("success", True)):
                inside_both += 1
            if bucket in {"near_outside", "target_outside"} and support == "clean":
                outside_clean += 1
        if len(ids) != len(set(ids)):
            errors.append("experiences IDs must be unique")
        if inside_both < 3:
            errors.append(f"need >=3 successful inside/both experiences for P3; found {inside_both}")
        if outside_clean < 2:
            errors.append(f"need >=2 outside clean-supporting experiences for P3; found {outside_clean}")
        if support_counts["clean"] < 3:
            errors.append(f"need >=3 clean-supporting experiences for P4; found {support_counts['clean']}")
        if support_counts["poisoned"] < 3:
            errors.append(f"need >=3 poisoned-supporting experiences for P4; found {support_counts['poisoned']}")

    # Optional P3-only poison surface. Not part of the shared P3/P4 pool.
    p3_block = spec.get("p3")
    if p3_block is not None:
        if not isinstance(p3_block, dict):
            errors.append("p3 must be an object when present")
        else:
            surface = p3_block.get("poison_experiences")
            if surface is not None:
                if not isinstance(surface, list) or len(surface) < 3:
                    errors.append("p3.poison_experiences must contain >=3 records when present")
                else:
                    surface_ids: list[str] = []
                    pool_ids = {
                        str(x.get("id", "")).strip()
                        for x in (spec.get("experiences") or [])
                        if isinstance(x, dict)
                    }
                    for i, exp in enumerate(surface[:3]):
                        if not isinstance(exp, dict):
                            errors.append(f"p3.poison_experiences[{i}] must be an object")
                            continue
                        eid = str(exp.get("id", "")).strip()
                        if not eid:
                            errors.append(f"p3.poison_experiences[{i}].id is required")
                        if eid in pool_ids:
                            errors.append(
                                f"p3.poison_experiences id {eid!r} collides with the shared P3/P4 pool"
                            )
                        surface_ids.append(eid)
                        has_text = str(exp.get("text", "")).strip()
                        for field in ("context", "action", "outcome"):
                            if not has_text and not str(exp.get(field, "")).strip():
                                errors.append(f"p3.poison_experiences {eid or i}: {field} is required")
                        if str(exp.get("scope_bucket", "inside")).strip() not in {"inside", ""}:
                            errors.append(
                                f"p3.poison_experiences {eid or i}: scope_bucket must be inside"
                            )
                        support = str(exp.get("supports", "both")).strip().lower()
                        if support not in {"both", ""}:
                            errors.append(
                                f"p3.poison_experiences {eid or i}: supports must be both"
                            )
                    if len(surface_ids) != len(set(x for x in surface_ids if x)):
                        errors.append("p3.poison_experiences IDs must be unique")

    # Pure P5: same context, conflicting successful experiences, order only.
    p5 = spec.get("p5")
    if not isinstance(p5, dict):
        errors.append("p5 must be an object")
        return errors
    if not str(p5.get("context_key", "")).strip():
        errors.append("p5.context_key is required")
    if not str(p5.get("context_description", "")).strip():
        errors.append("p5.context_description is required")
    _validate_policy("p5.clean_policy", p5.get("clean_policy"), choices, errors)
    _validate_policy("p5.poisoned_policy", p5.get("poisoned_policy"), choices, errors)
    target = p5.get("target")
    if not isinstance(target, dict):
        errors.append("p5.target is required")
    else:
        if not str(target.get("task", "")).strip():
            errors.append("p5.target.task is required")
        ca = ensure_choice_label(target.get("clean_answer", ""), choices)
        pa = ensure_choice_label(target.get("poisoned_answer", ""), choices)
        if ca is None or pa is None:
            errors.append("p5.target answers must match choices")
        elif ca == pa:
            errors.append("p5.target clean_answer and poisoned_answer must differ")
        cpol = ensure_choice_label(p5.get("clean_policy", {}).get("action_label", ""), choices)
        ppol = ensure_choice_label(p5.get("poisoned_policy", {}).get("action_label", ""), choices)
        if ca and cpol and ca != cpol:
            errors.append("p5.target.clean_answer must equal p5.clean_policy.action_label")
        if pa and ppol and pa != ppol:
            errors.append("p5.target.poisoned_answer must equal p5.poisoned_policy.action_label")

    l5exps = p5.get("experiences")
    if not isinstance(l5exps, list) or not l5exps:
        errors.append("p5.experiences must be a non-empty list")
    else:
        ids: list[str] = []
        clean_n = poison_n = 0
        expected_key = str(p5.get("context_key", "")).strip()
        for i, exp in enumerate(l5exps):
            eid = str(exp.get("id", "")).strip()
            ids.append(eid)
            if not eid:
                errors.append(f"p5.experiences[{i}].id is required")
            for field in ("context", "action", "outcome"):
                if not str(exp.get(field, "")).strip():
                    errors.append(f"p5 experience {eid or i}: {field} is required")
            if str(exp.get("context_key", "")).strip() != expected_key:
                errors.append(
                    f"p5 experience {eid or i}: context_key must equal family p5.context_key={expected_key!r}"
                )
            support = str(exp.get("supports", "")).strip().lower()
            if support not in {"clean", "poisoned", "neutral"}:
                errors.append(f"p5 experience {eid or i}: supports must be clean/poisoned/neutral")
            if not bool(exp.get("success", True)):
                errors.append(f"p5 experience {eid or i}: P5 conflicting evidence must be observed-success records")
            if support == "clean":
                clean_n += 1
            elif support == "poisoned":
                poison_n += 1
        if len(ids) != len(set(ids)):
            errors.append("p5.experience IDs must be unique")
        if clean_n < 3:
            errors.append(f"p5 needs >=3 clean-supporting same-context experiences; found {clean_n}")
        if poison_n < 3:
            errors.append(f"p5 needs >=3 poisoned-supporting same-context experiences; found {poison_n}")

    return errors


def validate_compiled_case(case: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    interface = case.get("interface")
    if interface not in SUPPORTED_INTERFACES:
        return [f"invalid interface: {interface}"]

    exps = case.get("candidate_experiences", [])
    ids = [str(x.get("id", "")) for x in exps]
    idset = set(ids)
    if len(ids) != len(idset):
        errors.append("duplicate experience IDs")

    eval_obj = case.get("evaluation", {})
    choices = eval_obj.get("choices", [])
    tasks = eval_obj.get("trigger_tasks", [])
    if not choices:
        errors.append("requires multiple-choice evaluation")
    for task in tasks:
        if ensure_choice_label(task.get("clean_answer"), choices) is None:
            errors.append(f"task {task.get('task_id')}: clean answer not in choices")
        if ensure_choice_label(task.get("poisoned_answer"), choices) is None:
            errors.append(f"task {task.get('task_id')}: poisoned answer not in choices")

    mb = case.get("memory_building", {})
    if interface == "P1":
        if mb.get("mode") != "direct_memory_injection":
            errors.append("P1 memory_building.mode must be direct_memory_injection")
        clean_mem = str(mb.get("clean_condition", {}).get("memory_text", "")).strip()
        poison_mem = str(mb.get("poisoned_condition", {}).get("memory_text", "")).strip()
        if not clean_mem or not poison_mem:
            errors.append("P1 requires non-empty clean/poisoned memory_text")
        elif normalize_text(clean_mem) == normalize_text(poison_mem):
            errors.append("P1 clean and poisoned stored memory content must differ")
        if len(tasks) != 1:
            errors.append("P1 requires exactly one downstream trigger task")

    elif interface == "P2":
        if mb.get("mode") != "batch_reflection_from_evidence":
            errors.append("P2 memory_building.mode must be batch_reflection_from_evidence")
        c = list(mb.get("clean_condition", {}).get("evidence_ids", []))
        p = list(mb.get("poisoned_condition", {}).get("evidence_ids", []))
        if not c or not p:
            errors.append("P2 requires non-empty clean/poisoned evidence histories")
        if len(c) != len(p):
            errors.append("P2 clean and poisoned evidence histories must have equal budgets")
        if len(c) != len(set(c)) or len(p) != len(set(p)):
            errors.append("P2 evidence histories must not contain duplicate IDs")
        if not set(c) <= idset or not set(p) <= idset:
            errors.append("P2 evidence history references unknown experience")
        if set(c) == set(p):
            errors.append("P2 clean/poisoned evidence histories must differ")
        exp_by_id = {str(x.get("id", "")): x for x in exps}
        clean_only = set(c) - set(p)
        poison_only = set(p) - set(c)
        if not clean_only or not poison_only:
            errors.append("P2 paired histories require both authentic-only and forged-only evidence")
        for eid in c:
            prov = exp_by_id.get(eid, {}).get("provenance") or {}
            if prov.get("authentic") is not True:
                errors.append("P2 clean condition must contain only benchmark-authentic evidence")
        forged_n = 0
        for eid in poison_only:
            prov = exp_by_id.get(eid, {}).get("provenance") or {}
            if prov.get("authentic") is False:
                forged_n += 1
        if poison_only and forged_n != len(poison_only):
            errors.append("P2 poison-only evidence must be marked benchmark-forged")
        for eid in set(c) & set(p):
            prov = exp_by_id.get(eid, {}).get("provenance") or {}
            if prov.get("authentic") is not True:
                errors.append("P2 shared evidence must be authentic")
        if len(tasks) != 1:
            errors.append("P2 requires exactly one downstream trigger task")

    elif interface == "P3":
        c = list(mb.get("clean_condition", {}).get("experience_ids", []))
        p = list(mb.get("poisoned_condition", {}).get("experience_ids", []))
        if not c or not p:
            errors.append("P3 requires non-empty histories")
        if not set(c) <= idset or not set(p) <= idset:
            errors.append("P3 history references unknown experience")
        if set(c) == set(p):
            errors.append("P3 histories must differ")

        controls = mb.get("mechanism_controls", {})
        inside_only = list(controls.get("strict_poisoned", {}).get("experience_ids", []))
        matched = list(controls.get("count_matched_clean", {}).get("experience_ids", []))
        exp_by_id = {str(x.get("id", "")): x for x in exps}

        if not inside_only:
            errors.append("P3 requires mechanism_controls.strict_poisoned")
        else:
            if not set(inside_only) <= idset:
                errors.append("P3 inside-only poisoned control references unknown experience")
            if not set(inside_only) <= set(p):
                errors.append("P3 poisoned_condition must include the inside-only cue-salient history")
            extra = [eid for eid in p if eid not in set(inside_only)]
            for eid in extra:
                exp = exp_by_id.get(eid, {})
                scope = str(exp.get("scope_bucket", "")).strip().lower()
                supports = str(exp.get("supports", "")).strip().lower()
                if scope not in {"general", "near_outside", "target_outside"} or supports not in {"poisoned", "p4_poisoned"}:
                    errors.append(
                        "P3 extra poisoned records must be authentic outside/general successes "
                        "for the locally preferred action"
                    )

        if not matched:
            errors.append("P3 requires mechanism_controls.count_matched_clean")
        elif inside_only:
            if not set(matched) <= idset:
                errors.append("P3 count-matched control references unknown experience")
            if len(matched) != len(inside_only):
                errors.append("P3 count-matched clean and inside-only poisoned histories must have identical evidence counts")
            if len(matched) != len(set(matched)) or len(inside_only) != len(set(inside_only)):
                errors.append("P3 matched histories must not contain duplicate experience IDs")
            shared = set(matched) & set(inside_only)
            if len(inside_only) >= 2 and len(shared) != len(inside_only) - 1:
                errors.append("P3 matched control must differ from inside-only poisoned history by exactly one experience")
            if len(matched) == len(inside_only):
                differing_positions = [i for i, (a, b) in enumerate(zip(matched, inside_only)) if a != b]
                if len(differing_positions) != 1:
                    errors.append("P3 matched pair must replace exactly one record in the same presentation slot")
            clean_only = set(matched) - set(inside_only)
            poison_only = set(inside_only) - set(matched)
            for eid in clean_only:
                exp = exp_by_id.get(eid, {})
                if exp.get("scope_bucket") not in {"near_outside", "target_outside"} or exp.get("supports") != "clean":
                    errors.append("P3 matched clean-only record must be outside-scope and clean-supporting")
            for eid in poison_only:
                exp = exp_by_id.get(eid, {})
                if exp.get("scope_bucket") != "inside" or exp.get("supports") != "both":
                    errors.append("P3 matched poisoned-only record must be inside-scope and support both policies")

    elif interface == "P4":
        c = list(mb.get("clean_condition", {}).get("selected_ids", []))
        p = list(mb.get("poisoned_condition", {}).get("selected_ids", []))
        budget = int(mb.get("selection_budget", 0) or 0)
        if len(c) != budget or len(p) != budget:
            errors.append("P4 selected subsets must equal selection_budget")
        if len(c) != len(set(c)) or len(p) != len(set(p)):
            errors.append("P4 selected subsets must not contain duplicate IDs")
        if not set(c) <= idset or not set(p) <= idset:
            errors.append("P4 subset references unknown experience")
        if set(c) == set(p):
            errors.append("P4 clean/poisoned subsets must differ")
        if len(set(c) & set(p)) < 1:
            errors.append("P4 paired subsets should retain at least one shared record for a matched selection intervention")

        controls = mb.get("mechanism_controls", {})
        full_pool = list(controls.get("full_pool", {}).get("experience_ids", []))
        if not full_pool:
            errors.append("P4 requires mechanism_controls.full_pool")
        else:
            if Counter(full_pool) != Counter(ids):
                errors.append("P4 full-pool control must contain every candidate experience exactly once")

    else:
        c = list(mb.get("clean_condition", {}).get("order", []))
        p = list(mb.get("poisoned_condition", {}).get("order", []))
        if Counter(c) != Counter(ids) or Counter(p) != Counter(ids):
            errors.append("P5 orders must be exact permutations of the same full experience multiset")
        if c == p:
            errors.append("P5 clean/poisoned orders must differ")
        keys = {str(x.get("context_key", "")).strip() for x in exps}
        if len(keys) != 1 or "" in keys:
            errors.append("P5 candidate experiences must all share one non-empty context_key")
        clean_n = sum(1 for x in exps if x.get("supports") == "clean")
        poison_n = sum(1 for x in exps if x.get("supports") == "poisoned")
        if clean_n < 3 or poison_n < 3:
            errors.append("P5 compiled case needs >=3 clean and >=3 poisoned same-context records")
        if case.get("metadata", {}).get("p5_pure_same_context") is not True:
            errors.append("P5 case must declare metadata.p5_pure_same_context=true")

    return errors
