"""JSPLIB job-shop loader → generic disjunctive makespan MILP (task 34).

Parses the standard `JSPLIB <https://github.com/tamy0612/JSPLIB>`_ / OR-Library
job-shop layout and maps it to the OPOP IR via the textbook disjunctive
(big-``M``) makespan model — applied uniformly to every instance:

* continuous start times ``s_{j,o} in [0, M]`` and a makespan ``Cmax in [0, M]``
  (``M = sum of all processing times``);
* precedence rows ``s_{j,o} - s_{j,o+1} <= -p_{j,o}`` (operation order within a job);
* makespan rows ``s_{j,last} - Cmax <= -p_{j,last}``;
* disjunctive big-``M`` rows for every pair of operations sharing a machine,
  gated by a binary order variable ``y`` (exactly one of the pair precedes);
* objective ``minimise Cmax``.

Layout (a free-form integer stream; ``#`` / blank lines are ignored)::

    n_jobs n_machines
    m p m p ...        # job 0: (machine, duration) pairs in process order
    ...

A truncated file, a non-integer token, or an out-of-range machine id raises
:class:`~opop.bench.classic.base.ParseError` with file + line context.
"""

from __future__ import annotations

from pathlib import Path

from opop.bench.classic.base import (
    ClassicAdapter,
    ParseError,
    TokenCursor,
    read_text,
    tag_instance,
)
from opop.model.adapter import register_adapter
from opop.model.ir import (
    MILP,
    ConstraintSense,
    LinearConstraint,
    Objective,
    ObjSense,
    Variable,
    VarType,
)

__all__ = ["ADAPTER", "load", "loads"]

#: The registered classic-CO adapter for the JSP family.
ADAPTER = ClassicAdapter(family="jsp", problem_class="JSP")


def _strip_comments(text: str) -> str:
    """Drop ``#``-prefixed comment lines (JSPLIB instances often carry a header)."""
    kept = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    return "\n".join(kept)


def loads(text: str, *, name: str = "jsp", source: str = "<string>") -> MILP:
    """Parse a JSPLIB ``text`` into a disjunctive makespan :class:`~opop.model.ir.MILP`.

    Raises:
        ParseError: On a missing header, a truncated job row, or an out-of-range
            machine id.
    """
    cursor = TokenCursor(_strip_comments(text), source=source)
    n_jobs = cursor.next_int("job count")
    n_machines = cursor.next_int("machine count")
    if n_jobs < 1 or n_machines < 1:
        raise ParseError(
            "job shop needs n_jobs >= 1 and n_machines >= 1, got "
            + f"{n_jobs}, {n_machines}",
            source=source,
            line=cursor.line,
        )

    jobs: list[list[tuple[int, float]]] = []
    for j in range(n_jobs):
        ops: list[tuple[int, float]] = []
        for o in range(n_machines):
            machine = cursor.next_int(f"machine of job {j} op {o}")
            duration = cursor.next_int(f"duration of job {j} op {o}")
            if not (0 <= machine < n_machines):
                raise ParseError(
                    f"job {j} op {o} machine {machine} out of range [0, {n_machines})",
                    source=source,
                    line=cursor.line,
                )
            ops.append((machine, float(duration)))
        jobs.append(ops)

    big_m = sum(dur for ops in jobs for _m, dur in ops) or 1.0

    variables: list[Variable] = [Variable("Cmax", VarType.CONTINUOUS, 0.0, big_m)]
    for j, ops in enumerate(jobs):
        variables.extend(
            Variable(f"s_{j}_{o}", VarType.CONTINUOUS, 0.0, big_m) for o in range(len(ops))
        )

    constraints: list[LinearConstraint] = []
    for j, ops in enumerate(jobs):
        for o in range(len(ops) - 1):
            constraints.append(
                LinearConstraint(
                    f"prec_{j}_{o}",
                    {f"s_{j}_{o}": 1.0, f"s_{j}_{o + 1}": -1.0},
                    ConstraintSense.LE,
                    -ops[o][1],
                )
            )
        last = len(ops) - 1
        constraints.append(
            LinearConstraint(
                f"makespan_{j}",
                {f"s_{j}_{last}": 1.0, "Cmax": -1.0},
                ConstraintSense.LE,
                -ops[last][1],
            )
        )

    by_machine: dict[int, list[tuple[int, int, float]]] = {}
    for j, ops in enumerate(jobs):
        for o, (machine, dur) in enumerate(ops):
            by_machine.setdefault(machine, []).append((j, o, dur))

    disjunctive_vars: list[Variable] = []
    for machine in sorted(by_machine):
        ops_on_m = by_machine[machine]
        for a in range(len(ops_on_m)):
            for b in range(a + 1, len(ops_on_m)):
                ja, oa, pa = ops_on_m[a]
                jb, ob, pb = ops_on_m[b]
                yname = f"y_{ja}_{oa}_{jb}_{ob}"
                disjunctive_vars.append(Variable(yname, VarType.BINARY, 0.0, 1.0))
                sa, sb = f"s_{ja}_{oa}", f"s_{jb}_{ob}"
                constraints.append(
                    LinearConstraint(
                        f"disj_{ja}_{oa}_{jb}_{ob}_ab",
                        {sa: 1.0, sb: -1.0, yname: big_m},
                        ConstraintSense.LE,
                        big_m - pa,
                    )
                )
                constraints.append(
                    LinearConstraint(
                        f"disj_{ja}_{oa}_{jb}_{ob}_ba",
                        {sb: 1.0, sa: -1.0, yname: -big_m},
                        ConstraintSense.LE,
                        -pb,
                    )
                )
    variables.extend(disjunctive_vars)

    objective = Objective(coeffs={"Cmax": 1.0}, sense=ObjSense.MINIMIZE)
    ir = MILP(
        name=name,
        variables=tuple(variables),
        constraints=tuple(constraints),
        objective=objective,
        metadata={
            "domain": "scheduling",
            "formulation": "disjunctive",
            "n_jobs": n_jobs,
            "n_machines": n_machines,
        },
    )
    return tag_instance(ir, family="jsp", source="jsplib", instance=name)


def load(path: str) -> MILP:
    """Load a JSPLIB ``.txt`` file into a disjunctive makespan :class:`~opop.model.ir.MILP`."""
    return loads(read_text(path), name=Path(path).stem, source=str(path))


register_adapter(ADAPTER)
