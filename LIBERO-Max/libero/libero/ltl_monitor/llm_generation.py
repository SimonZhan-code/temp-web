"""LLM-assisted LTL constraint generation for LIBERO tasks.

This module builds a static prompt payload from each task's BDDL file:

- task description / language instruction
- goal atomic propositions already derived locally from the BDDL goal
- safety atomic propositions derived locally for SafeLIBERO-style tasks

The LLM is only asked to infer:

- ordering-aware task formulas when a plain goal conjunction is insufficient
- safety formulas over the provided safety atomic propositions

Goal-success formulas remain locally generated from the BDDL goal predicates.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from http.client import IncompleteRead, RemoteDisconnected
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional

try:
    from libero.libero.benchmark.family import get_libero_suite_spec
    from libero.libero.ltl_monitor.builder import LDBABuildError, build_ldba
    from libero.libero.ltl_monitor.temporal_monitor import CONSTANTS, tokenize
except ImportError:
    from libero.benchmark.family import get_libero_suite_spec
    from libero.ltl_monitor.builder import LDBABuildError, build_ldba
    from libero.ltl_monitor.temporal_monitor import CONSTANTS, tokenize


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

SYSTEM_PROMPT = """You generate a single LTL formula that fully specifies a LIBERO manipulation task.

Rules:
- Return valid JSON only. No markdown fences.
- Use only atomic proposition names listed in `all_atomic_propositions`.
- Each atomic proposition carries a `category`: `goal` entries describe success conditions; `safety_violation` entries describe states that must be avoided.
- Do not invent new atomic propositions.
- The formula must eventually satisfy every `goal` atomic proposition, capturing any necessary temporal ordering with F or U.
- If `safety_violation` atomic propositions are present, the formula must globally avoid them — typically with a `G(!( ... ))` conjunct combined with the goal portion via `&`.
- Stay within the operator set: G, F, U, !, &, |, (, ).

Return exactly this schema:
{
  "ltl_formula": string,
  "description": string,
  "notes": [string, ...]
}
"""


def robosuite_parse_problem(problem_filename: str):
    with open(problem_filename, "r", encoding="utf-8") as handle:
        text = handle.read()
    parsed = _parse_sexpr(text)
    return _extract_problem_fields(parsed)


def _strip_line_comments(text: str) -> str:
    stripped_lines = []
    for line in text.splitlines():
        comment_idx = line.find(";")
        if comment_idx >= 0:
            line = line[:comment_idx]
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


def _tokenize_sexpr(text: str) -> list[str]:
    text = _strip_line_comments(text)
    return text.replace("(", " ( ").replace(")", " ) ").split()


def _parse_tokens(tokens: list[str], start: int = 0) -> tuple[Any, int]:
    token = tokens[start]
    if token == "(":
        out = []
        idx = start + 1
        while tokens[idx] != ")":
            item, idx = _parse_tokens(tokens, idx)
            out.append(item)
        return out, idx + 1
    if token == ")":
        raise ValueError("Unexpected closing parenthesis while parsing BDDL.")
    return token, start + 1


def _parse_sexpr(text: str) -> list[Any]:
    tokens = _tokenize_sexpr(text)
    expr, next_idx = _parse_tokens(tokens, 0)
    if next_idx != len(tokens):
        raise ValueError("Unexpected trailing tokens while parsing BDDL.")
    if not isinstance(expr, list):
        raise ValueError("Expected top-level list in BDDL file.")
    return expr


def _extract_typed_instances(items: list[Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    pending: list[str] = []
    idx = 0
    while idx < len(items):
        token = items[idx]
        if token == "-":
            type_name = str(items[idx + 1])
            grouped.setdefault(type_name, []).extend(str(x) for x in pending)
            pending = []
            idx += 2
            continue
        pending.append(str(token))
        idx += 1
    if pending:
        grouped.setdefault("object", []).extend(pending)
    return grouped


def _extract_goal_predicates(expr: Any) -> list[tuple[str, ...]]:
    if not isinstance(expr, list) or not expr:
        return []
    head = str(expr[0]).lower()
    if head == "and":
        out: list[tuple[str, ...]] = []
        for child in expr[1:]:
            out.extend(_extract_goal_predicates(child))
        return out
    return [tuple(str(token) for token in expr)]


def _extract_problem_fields(parsed: list[Any]) -> dict[str, Any]:
    if not parsed or parsed[0] != "define":
        raise ValueError("Expected BDDL file to start with define.")
    result = {
        "problem_name": "unknown",
        "objects": {},
        "obj_of_interest": [],
        "goal_state": [],
        "language_instruction": [],
    }
    for group in parsed[1:]:
        if not isinstance(group, list) or not group:
            continue
        tag = group[0]
        if tag == "problem":
            result["problem_name"] = str(group[-1])
        elif tag == ":objects":
            result["objects"] = _extract_typed_instances(group[1:])
        elif tag == ":obj_of_interest":
            result["obj_of_interest"] = [str(token) for token in group[1:]]
        elif tag == ":language":
            result["language_instruction"] = [str(token) for token in group[1:]]
        elif tag == ":goal" and len(group) > 1:
            result["goal_state"] = _extract_goal_predicates(group[1])
    return result


@dataclass(frozen=True)
class PromptAtomicProposition:
    name: str
    category: str
    description: str


@dataclass(frozen=True)
class TaskGenerationContext:
    task_id: str
    suite_name: str
    source: str
    taxonomy_level_1: str
    task_language: str
    objects_of_interest: tuple[str, ...]
    goal_formula_local: str
    goal_atomic_propositions: tuple[PromptAtomicProposition, ...]
    safety_atomic_propositions: tuple[PromptAtomicProposition, ...]
    raw_goal_predicates: tuple[str, ...]

    @property
    def all_atomic_propositions(self) -> tuple[PromptAtomicProposition, ...]:
        return self.goal_atomic_propositions + self.safety_atomic_propositions

    def as_prompt_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "suite_name": self.suite_name,
            "source": self.source,
            "taxonomy_level_1": self.taxonomy_level_1,
            "task_description": self.task_language,
            "objects_of_interest": list(self.objects_of_interest),
            "all_atomic_propositions": [asdict(prop) for prop in self.all_atomic_propositions],
        }


def _flatten_instance_names(grouped_instances: dict[str, list[str]]) -> list[str]:
    names: list[str] = []
    for instance_names in grouped_instances.values():
        names.extend(str(name) for name in instance_names)
    return names


def _normalize_goal_predicate_name(pred: str) -> str:
    return str(pred).lower()


def _format_goal_prop_name(goal_predicate: tuple[Any, ...], idx: int) -> str:
    tokens = [str(x) for x in goal_predicate]
    if not tokens:
        base = f"goal_{idx}"
    else:
        pred = _normalize_goal_predicate_name(tokens[0])
        if len(tokens) == 2:
            base = f"{pred}_{tokens[1]}"
        elif len(tokens) >= 3:
            base = f"{pred}_{tokens[1]}_{tokens[2]}"
        else:
            base = "_".join(tokens)
    return (
        base.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .lower()
    )


def _format_goal_prop_description(goal_predicate: tuple[Any, ...]) -> str:
    tokens = [str(x) for x in goal_predicate]
    if not tokens:
        return ""
    return f"{tokens[0]}({', '.join(tokens[1:])})"


def _make_goal_atomic_propositions(
    parsed_problem: dict[str, Any],
) -> tuple[PromptAtomicProposition, ...]:
    props: list[PromptAtomicProposition] = []
    seen_names: set[str] = set()
    for idx, goal_predicate in enumerate(parsed_problem.get("goal_state", [])):
        base_name = _format_goal_prop_name(tuple(goal_predicate), idx)
        name = base_name
        if name in seen_names:
            name = f"{base_name}_goal_{idx}"
        seen_names.add(name)
        props.append(
            PromptAtomicProposition(
                name=name,
                category="goal",
                description=_format_goal_prop_description(tuple(goal_predicate)),
            )
        )
    return tuple(props)


def _make_safety_atomic_propositions(
    parsed_problem: dict[str, Any],
    *,
    include_safety: bool,
) -> tuple[PromptAtomicProposition, ...]:
    if not include_safety:
        return ()
    movable_objects = _flatten_instance_names(parsed_problem.get("objects", {}))
    return tuple(
        PromptAtomicProposition(
            name=f"{obj_name}_displaced",
            category="safety_violation",
            description=f"displaced({obj_name}) > safety_threshold",
        )
        for obj_name in movable_objects
    )


def _make_goal_formula_local(goal_props: Iterable[PromptAtomicProposition]) -> str:
    names = [prop.name for prop in goal_props]
    if not names:
        return "true"
    return f"F({' & '.join(names)})"


def build_task_generation_context(
    *,
    suite_name: str,
    task_id: str,
    bddl_file_path: str,
) -> TaskGenerationContext:
    suite_spec = get_libero_suite_spec(suite_name)
    parsed_problem = robosuite_parse_problem(bddl_file_path)
    goal_atomic_props = _make_goal_atomic_propositions(parsed_problem)
    safety_atomic_props = _make_safety_atomic_propositions(
        parsed_problem,
        include_safety=suite_spec.source == "safety",
    )
    goal_formula_local = _make_goal_formula_local(goal_atomic_props)
    raw_goal_predicates = tuple(
        _format_goal_prop_description(tuple(goal_predicate))
        for goal_predicate in parsed_problem.get("goal_state", [])
    )
    language_tokens = parsed_problem.get("language_instruction", [])
    task_language = " ".join(str(tok) for tok in language_tokens)
    return TaskGenerationContext(
        task_id=task_id,
        suite_name=suite_name,
        source=suite_spec.source,
        taxonomy_level_1=suite_spec.taxonomy_level_1,
        task_language=task_language,
        objects_of_interest=tuple(
            str(obj_name) for obj_name in parsed_problem.get("obj_of_interest", [])
        ),
        goal_formula_local=goal_formula_local,
        goal_atomic_propositions=goal_atomic_props,
        safety_atomic_propositions=safety_atomic_props,
        raw_goal_predicates=raw_goal_predicates,
    )


def iter_suite_generation_contexts(suite_name: str) -> list[TaskGenerationContext]:
    suite_spec = get_libero_suite_spec(suite_name)
    contexts: list[TaskGenerationContext] = []
    for task in suite_spec.tasks:
        bddl_file_path = os.path.join(
            suite_spec.bddl_root,
            task.problem_folder,
            task.bddl_file,
        )
        contexts.append(
            build_task_generation_context(
                suite_name=suite_name,
                task_id=task.name,
                bddl_file_path=bddl_file_path,
            )
        )
    return contexts


def build_generation_messages(context: TaskGenerationContext) -> list[dict[str, str]]:
    payload = json.dumps(context.as_prompt_payload(), indent=2, sort_keys=True)
    user_prompt = (
        "Generate ordering and safety LTL constraints for this task.\n\n"
        "Task payload:\n"
        f"{payload}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", stripped)
        stripped = re.sub(r"\n```$", "", stripped)
        stripped = stripped.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


class _FormulaParser:
    def __init__(self, formula: str, *, allowed_atoms: set[str]):
        self.formula = formula
        self.tokens = tokenize(formula)
        self.allowed_atoms = set(allowed_atoms)
        self.index = 0

        normalized_input = re.sub(r"\s+", "", formula)
        normalized_tokens = "".join(self.tokens)
        if normalized_input != normalized_tokens:
            raise ValueError(f"Formula contains unsupported syntax: {formula}")
        if not self.tokens:
            raise ValueError("Formula is empty.")

    def parse(self) -> None:
        self._parse_or()
        if self.index != len(self.tokens):
            raise ValueError(
                f"Unexpected trailing tokens in formula '{self.formula}': "
                f"{self.tokens[self.index:]}"
            )

    def _peek(self) -> Optional[str]:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]

    def _consume(self, expected: Optional[str] = None) -> str:
        token = self._peek()
        if token is None:
            raise ValueError(f"Unexpected end of formula: {self.formula}")
        if expected is not None and token != expected:
            raise ValueError(
                f"Expected token '{expected}' but found '{token}' in formula '{self.formula}'"
            )
        self.index += 1
        return token

    def _parse_or(self) -> None:
        self._parse_and()
        while self._peek() == "|":
            self._consume("|")
            self._parse_and()

    def _parse_and(self) -> None:
        self._parse_until()
        while self._peek() == "&":
            self._consume("&")
            self._parse_until()

    def _parse_until(self) -> None:
        self._parse_unary()
        while self._peek() == "U":
            self._consume("U")
            self._parse_unary()

    def _parse_unary(self) -> None:
        token = self._peek()
        if token in {"!", "F", "G", "X"}:
            self._consume()
            self._parse_unary()
            return
        if token == "(":
            self._consume("(")
            self._parse_or()
            self._consume(")")
            return
        self._parse_atom()

    def _parse_atom(self) -> None:
        token = self._consume()
        if token.lower() in CONSTANTS:
            return
        if token not in self.allowed_atoms:
            raise ValueError(
                f"Unknown atomic proposition '{token}' in formula '{self.formula}'. "
                f"Allowed atoms: {sorted(self.allowed_atoms)}"
            )


def validate_ltl_formula_syntax(formula: str, *, allowed_atoms: set[str]) -> None:
    """Validate a restricted LTL formula over a fixed AP vocabulary.

    This mirrors SENTINEL's approach: parse a constrained grammar locally and
    reject unsupported syntax with a hard error before evaluation.
    """

    _FormulaParser(formula, allowed_atoms=allowed_atoms).parse()


def validate_generated_constraints(
    context: TaskGenerationContext,
    *,
    ltl_formula: str,
) -> None:
    if not ltl_formula:
        raise ValueError("LLM did not return a non-empty ltl_formula.")
    allowed_atoms = {prop.name for prop in context.all_atomic_propositions}
    validate_ltl_formula_syntax(ltl_formula, allowed_atoms=allowed_atoms)
    _validate_ltl_semantics(ltl_formula, allowed_atoms)


def _validate_ltl_semantics(formula: str, propositions: set[str]) -> None:
    """Compile the formula through Rabinizer and require a finite specification.

    This is the semantic gate above syntax + vocabulary: an LDBA whose only
    accepting SCC is a singleton bottom SCC, meaning the spec terminates on a
    bounded prefix rather than requiring an infinite trace. Gracefully skips
    (with a warning) when the LDBA pipeline isn't available — typical for
    CI without Rabinizer.
    """

    try:
        ldba = build_ldba(formula, propositions)
    except LDBABuildError as exc:
        warnings.warn(
            f"Skipping semantic LTL check for '{formula}': {exc}. "
            "Set RABINIZER_PATH or supply a cached HOA to enable the "
            "finite-specification check.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if not ldba.is_finite_specification():
        raise ValueError(
            f"LTL formula does not compile to a finite specification "
            f"(no single accepting bottom SCC): {formula}"
        )


class OpenRouterClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        app_name: str = "LIBERO-MAX-LTL-Generator",
        referer: str = "https://local.libero-max",
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        self.api_key = api_key or os.environ.get(OPENROUTER_API_KEY_ENV, "").strip()
        if not self.api_key:
            raise ValueError(
                f"Missing OpenRouter API key. Set {OPENROUTER_API_KEY_ENV} in the environment."
            )
        self.model = model
        self.app_name = app_name
        self.referer = referer
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)

    def generate_constraints(
        self,
        context: TaskGenerationContext,
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "messages": build_generation_messages(context),
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_API_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.referer,
                "X-Title": self.app_name,
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"OpenRouter request failed with HTTP {exc.code}: {error_body}"
                ) from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                IncompleteRead,
                RemoteDisconnected,
                ConnectionError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
                time.sleep(self.retry_backoff_seconds * attempt)
        else:
            raise RuntimeError(f"OpenRouter request failed: {last_error}")

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {payload}") from exc
        result = _extract_json_object(content)
        return compose_generation_record(context, result)


def compose_generation_record(
    context: TaskGenerationContext,
    llm_response: dict[str, Any],
) -> dict[str, Any]:
    ltl_formula = llm_response.get("ltl_formula")
    validate_generated_constraints(context, ltl_formula=ltl_formula)
    return {
        "task_id": context.task_id,
        "suite_name": context.suite_name,
        "source": context.source,
        "taxonomy_level_1": context.taxonomy_level_1,
        "task_description": context.task_language,
        "goal_formula_local": context.goal_formula_local,
        "ltl_formula": ltl_formula,
        "description": llm_response.get("description", ""),
        "notes": list(llm_response.get("notes", [])),
        "all_atomic_propositions": [
            asdict(prop) for prop in context.all_atomic_propositions
        ],
    }
