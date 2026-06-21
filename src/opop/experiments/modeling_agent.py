"""LLM modeling-agent-only baseline: NL -> model -> solve (plan task 37, baseline 4).

A faithful, dependency-light reproduction of the OptiMUS / LLMOPT / ORLM / OR-R1
family of *modeling agents*: read a natural-language description of an
optimization problem, prompt an LLM to emit a precise MILP formulation as a
structured JSON object, parse that JSON into the symbolic :class:`opop.model.ir.MILP`
IR, solve it with an open-source solver kernel, and score the resulting trace.

This baseline deliberately runs the ``formulate -> (repair) -> build -> solve ->
evaluate`` pipeline ONLY. It is **not** opop: it never invokes the structural
analyzer, the verification gate, or the Bayesian controller / ask-tell loop.
That separation is what makes the T4 novelty claim ("analyzer-certified deltas
significant beyond modeling-agent-only") measurable, so it is enforced both by
construction (this module imports none of those packages) and by the
:data:`MODELING_AGENT_PHASES` / :data:`FORBIDDEN_LOOP_PHASES` pipeline tags a
caller can inspect.

The LLM is fully swappable (:class:`opop.llm.client.LLMClient`):
:class:`opop.llm.client.FakeLLMClient` gives deterministic, offline, network-free
runs for tests, while :class:`opop.llm.client.OpenAICompatClient` drives a real
model. A bounded self-correction (repair) loop re-prompts the LLM with the build
error when its first formulation is malformed — the OptiMUS/LLMOPT debugging
step — so the baseline is not under-resourced relative to opop's per-iteration
LLM calls.

The JSON model schema the agent expects (and :func:`milp_to_spec` emits)::

    {
      "name": "<short name>",
      "sense": "minimize" | "maximize",
      "variables": [
        {"name": "x", "type": "binary"|"integer"|"continuous",
         "lower": <number, optional>, "upper": <number, optional>}
      ],
      "objective": {"x": <coeff>, ...},
      "objective_offset": <number, optional>,
      "constraints": [
        {"name": "c", "coeffs": {"x": <coeff>, ...},
         "sense": "<=" | ">=" | "=", "rhs": <number>}
      ]
    }
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from opop.evaluator import evaluate
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)
from opop.model.state import Phi
from opop.solver.scip import ScipKernel

if TYPE_CHECKING:
    from opop.llm.client import LLMClient
    from opop.model.state import ScoreRecord, SolveTrace
    from opop.solver.kernel import SolverKernel

logger = logging.getLogger(__name__)

__all__ = [
    "FORBIDDEN_LOOP_PHASES",
    "MODELING_AGENT_PHASES",
    "PHASE_BUILD",
    "PHASE_EVALUATE",
    "PHASE_FORMULATE",
    "PHASE_REPAIR",
    "PHASE_SOLVE",
    "SYSTEM_PROMPT",
    "ModelSpecError",
    "ModelingAgentResult",
    "build_milp_from_spec",
    "build_prompt",
    "describe_milp",
    "milp_to_spec",
    "run_modeling_agent",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ModelSpecError(ValueError):
    """Raised when an LLM-emitted model spec cannot be built into a valid MILP.

    A subclass of :class:`ValueError` so callers may catch either. The bounded
    repair loop in :func:`run_modeling_agent` catches this to re-prompt the LLM.
    """


# ---------------------------------------------------------------------------
# Pipeline phase tags (the ONLY phases this baseline runs)
# ---------------------------------------------------------------------------
PHASE_FORMULATE = "formulate"  #: LLM call: NL problem -> JSON model spec.
PHASE_REPAIR = "repair"  #: LLM call: re-prompt with the build error (self-correction).
PHASE_BUILD = "build"  #: Deterministic JSON spec -> MILP IR.
PHASE_SOLVE = "solve"  #: Open-source solver kernel solve.
PHASE_EVALUATE = "evaluate"  #: Metric evaluation of the solve trace.

#: Every phase the modeling-agent baseline may execute. A run's ``pipeline`` is
#: always a subset of this set.
MODELING_AGENT_PHASES: frozenset[str] = frozenset(
    {PHASE_FORMULATE, PHASE_REPAIR, PHASE_BUILD, PHASE_SOLVE, PHASE_EVALUATE}
)

#: opop closed-loop phases that must NEVER appear in this baseline's pipeline —
#: the structural analyzer, the verification gate, and the BO ask-tell loop.
FORBIDDEN_LOOP_PHASES: frozenset[str] = frozenset(
    {"analyze", "verify", "controller_ask", "controller_tell", "ask", "tell", "propose"}
)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert operations-research modeling agent (in the style of "
    "OptiMUS, LLMOPT, ORLM, and OR-R1). Read a natural-language description of a "
    "combinatorial / mixed-integer optimization problem and produce a precise "
    "mathematical MILP formulation as a SINGLE JSON object.\n\n"
    "Output ONLY a JSON object with this exact schema:\n"
    "{\n"
    '  "name": "<short problem name>",\n'
    '  "sense": "minimize" | "maximize",\n'
    '  "variables": [\n'
    '    {"name": "<var>", "type": "binary" | "integer" | "continuous",\n'
    '     "lower": <number, optional>, "upper": <number, optional>}\n'
    "  ],\n"
    '  "objective": {"<var>": <coefficient>, ...},\n'
    '  "objective_offset": <number, optional>,\n'
    '  "constraints": [\n'
    '    {"name": "<row>", "coeffs": {"<var>": <coefficient>, ...},\n'
    '     "sense": "<=" | ">=" | "=", "rhs": <number>}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Decide the variables, their domains, the objective sense, and every linear constraint.\n"
    "- Use only linear constraints with senses <=, >=, or =.\n"
    "- Every coefficient key MUST be a declared variable name.\n"
    "- Emit the JSON object only: no prose, no explanation, no markdown code fences.\n"
)


def build_prompt(problem: str) -> str:
    """Build the user prompt that asks the LLM to formulate ``problem`` as a MILP."""
    return (
        "Formulate the following optimization problem as a MILP and return the "
        + "JSON model:\n\n"
        + problem
    )


def _repair_prompt(problem: str, previous: str, error: str) -> str:
    """Build the self-correction prompt re-issued after a failed model build."""
    return (
        "The previous model you produced was invalid and could not be built.\n"
        + f"Build error: {error}\n\n"
        + "Previous output:\n"
        + previous
        + "\n\nRe-read the problem and emit a corrected JSON model (JSON object "
        + "only):\n\n"
        + problem
    )


# ---------------------------------------------------------------------------
# Spec <-> MILP IR
# ---------------------------------------------------------------------------
_SPEC_TYPE_TO_VTYPE: dict[str, VarType] = {
    "binary": VarType.BINARY,
    "bin": VarType.BINARY,
    "integer": VarType.INTEGER,
    "int": VarType.INTEGER,
    "continuous": VarType.CONTINUOUS,
    "cont": VarType.CONTINUOUS,
}
_SPEC_SENSE_TO_CONSTRAINT: dict[str, ConstraintSense] = {
    "<=": ConstraintSense.LE,
    "le": ConstraintSense.LE,
    ">=": ConstraintSense.GE,
    "ge": ConstraintSense.GE,
    "=": ConstraintSense.EQ,
    "==": ConstraintSense.EQ,
    "eq": ConstraintSense.EQ,
}
_SPEC_OBJ_SENSE: dict[str, ObjSense] = {
    "minimize": ObjSense.MINIMIZE,
    "min": ObjSense.MINIMIZE,
    "maximize": ObjSense.MAXIMIZE,
    "max": ObjSense.MAXIMIZE,
}


def _coerce_bound(value: Any, default: float) -> float:
    """Coerce a JSON bound to a float, accepting ``None`` and ``inf`` strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ModelSpecError(f"variable bound must be numeric, got bool {value!r}")
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("inf", "+inf", "infinity", "+infinity"):
            return math.inf
        if token in ("-inf", "-infinity"):
            return -math.inf
        try:
            return float(token)
        except ValueError as exc:
            raise ModelSpecError(f"invalid variable bound {value!r}") from exc
    if isinstance(value, (int, float)):
        return float(value)
    raise ModelSpecError(f"variable bound must be numeric, got {type(value).__name__}")


def _coerce_coeffs(raw: Any, *, where: str) -> dict[str, float]:
    """Coerce a JSON ``{var: coeff}`` mapping to ``dict[str, float]`` (fail-loud)."""
    if not isinstance(raw, Mapping):
        raise ModelSpecError(f"{where} coefficients must be a JSON object")
    coeffs: dict[str, float] = {}
    raw_map: Mapping[Any, Any] = raw
    for key, value in raw_map.items():
        if not isinstance(key, str):
            raise ModelSpecError(f"{where} coefficient key must be a string, got {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ModelSpecError(f"{where} coefficient for {key!r} must be numeric")
        coeffs[key] = float(value)
    return coeffs


def build_milp_from_spec(spec: Mapping[str, Any], *, name: str = "") -> MILP:
    """Parse an LLM-emitted JSON model ``spec`` into a :class:`MILP` IR.

    The objective sense, variables (with domains), objective coefficients, and
    linear constraints are validated; binary variables default to ``[0, 1]`` and
    other variables to ``[0, inf)`` when bounds are omitted. Any malformed field
    (missing variables, unknown vtype/sense, non-numeric coefficient, or a
    coefficient naming an undeclared variable) raises :class:`ModelSpecError`.

    Args:
        spec: The parsed JSON model spec (see the module docstring schema).
        name: Problem name for the resulting IR (falls back to ``spec["name"]``).

    Returns:
        A validated :class:`MILP` IR ready to solve.
    """
    sense_raw = spec.get("sense", "minimize")
    obj_sense = _SPEC_OBJ_SENSE.get(str(sense_raw).strip().lower())
    if obj_sense is None:
        raise ModelSpecError(
            f"unknown objective sense {sense_raw!r}; expected minimize/maximize"
        )

    raw_vars = spec.get("variables")
    if not isinstance(raw_vars, list) or not raw_vars:
        raise ModelSpecError("model spec must declare a non-empty 'variables' list")
    var_entries: list[Any] = raw_vars
    variables: list[Variable] = []
    for entry in var_entries:
        if not isinstance(entry, Mapping):
            raise ModelSpecError(f"each variable must be an object, got {entry!r}")
        vname = entry.get("name")
        if not isinstance(vname, str) or not vname:
            raise ModelSpecError(f"variable is missing a string 'name': {entry!r}")
        vtype = _SPEC_TYPE_TO_VTYPE.get(str(entry.get("type", "continuous")).strip().lower())
        if vtype is None:
            raise ModelSpecError(f"variable {vname!r} has unknown type {entry.get('type')!r}")
        default_upper = 1.0 if vtype is VarType.BINARY else math.inf
        lower = _coerce_bound(entry.get("lower"), 0.0)
        upper = _coerce_bound(entry.get("upper"), default_upper)
        variables.append(Variable(name=vname, vtype=vtype, lower=lower, upper=upper))

    objective = Objective(
        coeffs=_coerce_coeffs(spec.get("objective", {}), where="objective"),
        sense=obj_sense,
        offset=_coerce_bound(spec.get("objective_offset"), 0.0),
    )

    raw_cons = spec.get("constraints", [])
    if not isinstance(raw_cons, list):
        raise ModelSpecError("'constraints' must be a JSON array")
    con_entries: list[Any] = raw_cons
    constraints: list[LinearConstraint] = []
    for idx, entry in enumerate(con_entries):
        if not isinstance(entry, Mapping):
            raise ModelSpecError(f"each constraint must be an object, got {entry!r}")
        cname = entry.get("name")
        cname = str(cname) if isinstance(cname, str) and cname else f"c{idx}"
        sense_c = _SPEC_SENSE_TO_CONSTRAINT.get(str(entry.get("sense", "<=")).strip().lower())
        if sense_c is None:
            raise ModelSpecError(f"constraint {cname!r} has unknown sense {entry.get('sense')!r}")
        rhs = _coerce_bound(entry.get("rhs", 0.0), 0.0)
        constraints.append(
            LinearConstraint(
                name=cname,
                coeffs=_coerce_coeffs(entry.get("coeffs", {}), where=f"constraint {cname!r}"),
                sense=sense_c,
                rhs=rhs,
            )
        )

    model_name = name or (str(spec["name"]) if isinstance(spec.get("name"), str) else "")
    try:
        return MILP(
            name=model_name or "modeling_agent_model",
            variables=tuple(variables),
            constraints=tuple(constraints),
            objective=objective,
            metadata={"source": "modeling_agent"},
        )
    except ValueError as exc:
        # Referential-integrity failure (e.g. a coefficient names an undeclared
        # variable) — surface it as a spec error so the repair loop can react.
        raise ModelSpecError(f"model failed validation: {exc}") from exc


def _bound_to_spec(value: float) -> float | str:
    """Render a bound for the JSON spec (``+-inf`` as a string, else a float)."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return float(value)


def milp_to_spec(ir: MILP) -> dict[str, Any]:
    """Serialise a :class:`MILP` IR to the JSON model spec (inverse of build).

    Bounds are emitted only when they differ from the vtype default (binary
    ``[0, 1]``, otherwise ``[0, inf)``); infinite non-default bounds are written
    as ``"inf"`` / ``"-inf"`` strings so the spec stays plain JSON (no ``Infinity``
    token). ``build_milp_from_spec(milp_to_spec(ir))`` reproduces ``ir`` exactly.
    """
    variables: list[dict[str, Any]] = []
    for var in ir.variables:
        entry: dict[str, Any] = {"name": var.name, "type": var.vtype.value.lower()}
        default_upper = 1.0 if var.vtype is VarType.BINARY else math.inf
        if var.lower != 0.0:
            entry["lower"] = _bound_to_spec(var.lower)
        if var.upper != default_upper:
            entry["upper"] = _bound_to_spec(var.upper)
        variables.append(entry)

    spec: dict[str, Any] = {
        "name": ir.name,
        "sense": ir.objective.sense.value,
        "variables": variables,
        "objective": {key: float(coeff) for key, coeff in ir.objective.coeffs.items()},
        "constraints": [
            {
                "name": con.name,
                "coeffs": {key: float(coeff) for key, coeff in con.coeffs.items()},
                "sense": con.sense.value,
                "rhs": float(con.rhs),
            }
            for con in ir.constraints
        ],
    }
    if ir.objective.offset != 0.0:
        spec["objective_offset"] = float(ir.objective.offset)
    return spec


def describe_milp(ir: MILP) -> str:
    """Render a :class:`MILP` IR as a natural-language problem statement.

    This is the NL input handed to the modeling agent. With a real LLM it
    exercises genuine NL -> formulation; with a :class:`FakeLLMClient` the text
    is ignored in favour of the canned response, but it keeps the pipeline
    faithful (the agent always receives a textual problem, never the IR).
    """
    sense = "Maximize" if ir.objective.sense is ObjSense.MAXIMIZE else "Minimize"
    obj_terms = " + ".join(
        f"{coeff:g}*{name}" for name, coeff in ir.objective.coeffs.items()
    ) or "0"
    offset = f" + {ir.objective.offset:g}" if ir.objective.offset != 0.0 else ""
    lines = [
        f"Problem: {ir.name or 'optimization problem'}.",
        f"{sense} the objective {obj_terms}{offset}.",
        f"There are {ir.n_vars} decision variables and {ir.n_constraints} constraints.",
    ]
    by_type: dict[str, list[str]] = {}
    for var in ir.variables:
        by_type.setdefault(var.vtype.value.lower(), []).append(var.name)
    for vtype, names in by_type.items():
        lines.append(f"Variables {', '.join(names)} are {vtype}.")
    for con in ir.constraints:
        terms = " + ".join(f"{coeff:g}*{name}" for name, coeff in con.coeffs.items()) or "0"
        lines.append(f"Constraint {con.name}: {terms} {con.sense.value} {con.rhs:g}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def _loads_spec(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from raw LLM text (direct / ```json fence / braces)."""
    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    marker = "```json"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.find("```", start)
        if end != -1:
            candidates.append(text[start:end])
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{") : text.rindex("}") + 1])
    for candidate in candidates:
        try:
            parsed: Any = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _tracker_summary(llm: LLMClient) -> dict[str, Any]:
    """Normalise an LLM client's token tracker into flat cost columns."""
    tracker = getattr(llm, "tracker", None)
    if tracker is None:
        return {"calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    summary: dict[str, Any] = tracker.summary()
    return {
        "calls": int(summary.get("calls", 0)),
        "tokens_in": int(summary.get("total_tokens_in", 0)),
        "tokens_out": int(summary.get("total_tokens_out", 0)),
        "cost_usd": float(summary.get("total_cost_usd", 0.0)),
    }


def _formulate(llm: LLMClient, prompt: str) -> tuple[str, dict[str, Any] | None]:
    """Issue one LLM modeling call; return ``(raw_text, parsed_spec_or_None)``.

    ``chat`` (not ``chat_json``) is used so the raw text is retained for the
    repair prompt while token usage is still recorded on the client's tracker.
    """
    raw = llm.chat(prompt, system=SYSTEM_PROMPT)
    return raw, _loads_spec(raw)


# ---------------------------------------------------------------------------
# Result + driver
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ModelingAgentResult:
    """Outcome of one NL -> model -> solve modeling-agent run.

    Attributes:
        ir: The MILP the agent built (``None`` if modeling failed after repairs).
        trace: The solver trajectory (``None`` if the model was never built).
        score: The evaluated metric record (``None`` if never solved).
        pipeline: Ordered phase tags executed — always a subset of
            :data:`MODELING_AGENT_PHASES` and disjoint from
            :data:`FORBIDDEN_LOOP_PHASES`.
        llm_summary: Flat LLM cost summary (``calls``/``tokens_in``/``tokens_out``/
            ``cost_usd``).
        n_repairs: Number of self-correction LLM calls used.
        solved: Whether the solver certified optimality.
        error: The build error if modeling failed, else ``None``.
        timings: Per-phase wall-clock seconds (``formulate``/``solve``/``evaluate``)
            for the baseline harness's cost accounting.
    """

    ir: MILP | None
    trace: SolveTrace | None
    score: ScoreRecord | None
    pipeline: tuple[str, ...]
    llm_summary: dict[str, Any]
    n_repairs: int
    solved: bool
    error: str | None = None
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def n_llm_calls(self) -> int:
        """Number of LLM calls made (one formulate + one per repair)."""
        return 1 + int(self.n_repairs)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serialisable summary of the run."""
        metrics: dict[str, Any] = dict(self.score.metrics) if self.score is not None else {}
        return {
            "model": (
                {"name": self.ir.name, "n_vars": self.ir.n_vars, "n_constraints": self.ir.n_constraints}
                if self.ir is not None
                else None
            ),
            "pipeline": list(self.pipeline),
            "llm_summary": dict(self.llm_summary),
            "n_repairs": self.n_repairs,
            "n_llm_calls": self.n_llm_calls,
            "solved": self.solved,
            "error": self.error,
            "metrics": metrics,
            "timings": dict(self.timings),
        }


def run_modeling_agent(
    problem: str,
    *,
    llm: LLMClient,
    kernel: SolverKernel | None = None,
    time_limit: float = 30.0,
    memory_limit_mb: int = 4096,
    seed: int = 0,
    reference_optimum: float | None = None,
    instance_id: str = "",
    max_repairs: int = 1,
) -> ModelingAgentResult:
    """Run the modeling-agent baseline: NL ``problem`` -> MILP -> solve -> score.

    The pipeline is strictly ``formulate -> [repair...] -> build -> solve ->
    evaluate``; it NEVER calls the structural analyzer, the verification gate, or
    the Bayesian controller (this module does not even import them). When the
    first formulation fails to build, up to ``max_repairs`` self-correction LLM
    calls re-prompt the model with the build error.

    Args:
        problem: Natural-language problem statement (see :func:`describe_milp`).
        llm: The LLM client (swap a :class:`FakeLLMClient` in for offline tests).
        kernel: Solver kernel; defaults to a fresh :class:`ScipKernel` (never Gurobi).
        time_limit: Per-solve wall-clock limit (seconds).
        memory_limit_mb: Per-solve memory ceiling (MiB).
        seed: Solver seed.
        reference_optimum: Known optimum forwarded to the evaluator.
        instance_id: Identifier stamped onto the built IR / trace.
        max_repairs: Maximum self-correction LLM calls on a malformed model.

    Returns:
        A :class:`ModelingAgentResult`.
    """
    solver: SolverKernel = kernel if kernel is not None else ScipKernel()
    pipeline: list[str] = [PHASE_FORMULATE]
    timings: dict[str, float] = {"formulate": 0.0, "solve": 0.0, "evaluate": 0.0}

    t_formulate = time.monotonic()
    raw_text, spec = _formulate(llm, build_prompt(problem))
    timings["formulate"] += time.monotonic() - t_formulate

    ir: MILP | None = None
    last_error: str | None = None
    n_repairs = 0
    for attempt in range(max_repairs + 1):
        try:
            ir = build_milp_from_spec(spec if spec is not None else {}, name=instance_id)
            break
        except ModelSpecError as exc:
            last_error = str(exc)
            if attempt >= max_repairs:
                break
            n_repairs += 1
            pipeline.append(PHASE_REPAIR)
            t_repair = time.monotonic()
            raw_text, spec = _formulate(llm, _repair_prompt(problem, raw_text, last_error))
            timings["formulate"] += time.monotonic() - t_repair

    if ir is None:
        logger.warning("modeling agent failed to build a model: %s", last_error)
        return ModelingAgentResult(
            ir=None,
            trace=None,
            score=None,
            pipeline=tuple(pipeline),
            llm_summary=_tracker_summary(llm),
            n_repairs=n_repairs,
            solved=False,
            error=last_error or "model build failed",
            timings=timings,
        )
    pipeline.append(PHASE_BUILD)

    pipeline.append(PHASE_SOLVE)
    t_solve = time.monotonic()
    trace = solver.solve(
        ir,
        Phi(),
        time_limit=float(time_limit),
        memory_limit_mb=int(memory_limit_mb),
        seed=int(seed),
    )
    timings["solve"] = time.monotonic() - t_solve

    pipeline.append(PHASE_EVALUATE)
    t_evaluate = time.monotonic()
    score = evaluate(trace, reference_optimum=reference_optimum, time_limit=float(time_limit))
    timings["evaluate"] = time.monotonic() - t_evaluate

    return ModelingAgentResult(
        ir=ir,
        trace=trace,
        score=score,
        pipeline=tuple(pipeline),
        llm_summary=_tracker_summary(llm),
        n_repairs=n_repairs,
        solved=bool(score.metrics.get("optimal", 0.0)),
        error=None,
        timings=timings,
    )
