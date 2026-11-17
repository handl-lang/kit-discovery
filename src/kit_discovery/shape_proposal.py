"""Shape-aware law proposal — target the shapes real models contain.

The proposal retro (`project/retros/law-proposal.md`) enumerated an
*algebraic* grammar and found 11 true, cost-reducing laws; the impact
retro (`project/retros/law-impact.md`) then measured them on real
graphs and got **zero firings** — the ops were present, the *shapes*
were not.  Proposal was decoupled from reality: it optimised algebraic
novelty, not applicability.

This tool closes that loop.  It reads the shape census
(`kit_discovery.census`) and, for the shapes real models actually
contain, asks the only question that matters: **is there a true,
cost-reducing equality whose LHS is (or contains) that shape?**

For each *schema* — a generic metavariable equality such as
``add(mul(A,B), mul(A,C)) -> mul(A, add(B,C))`` — it measures:

* **real matches** — subterms of the corpus (bench law cases + the 22
  exported ``kit_torch.models`` blocks) that the schema's LHS
  matches *with its metavariable equalities enforced*.  A schema whose
  LHS never matches with the equalities is inapplicable, however true
  it is.
* **relaxed matches** — the same count with repeated metavariables
  relaxed to wildcards (the pure *shape*, no precondition).  The gap
  between the two is exactly "the shape is present, the equality is
  not" — the impact retro's finding, made per-schema.
* **true** — the instantiated LHS and RHS agree numerically on random
  fp64 tensors of the *real* leaf shapes (the numeric oracle
  ``law_proposal`` uses for new axioms).
* **derivable** — ``kit_discovery.verifier.verify_law`` proves it from
  ``ALL_RULES`` (then it is a composite and cannot lower a cost).
* **new** — structurally not a duplicate or inverse of a library law.
* **useful** — saturating a real matched term under ``ALL_RULES`` vs
  ``ALL_RULES + {schema}`` strictly lowers the extracted cost.
* **fires / pays** — running the schema *alone* over every real model,
  how many fire and whether the extracted cost drops (verified through
  the pipeline's own ``_lower_extracted`` + ``sink.verify``).

The schemas are defined *inside this tool* — nothing is added to
``packages/`` or the shipped ``ALL_RULES``.  A firing table of zeros
is a decisive answer, not a bug.

Run::

    .venv/bin/python -m kit_discovery.shape_proposal
    .venv/bin/python -m kit_discovery.shape_proposal --json /tmp/shape.json

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from mlopt_core.cost import dag_cost
from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.egraph.terms import _term_instantiate, _term_match
from mlopt_core.ir import Op
from mlopt_core.laws import ALL_RULES
from mlopt_core.typing import _shape_of

# Sibling tools: the census/corpus, the verifier, and the numeric
# oracle + saturation helpers the proposal retro already built.
from kit_discovery import lawdata
from kit_discovery.impact import (
    _bench_cases,
    _cost_fn,
    _iter_subterms,
    model_cases,
    new_laws,
    reach_row,
)
from kit_discovery.proposal import (
    _library_keys,
    _numeric_true,
    _relation,
    _sat_cost,
    _spec_term,
)
from kit_discovery.verifier import verify_law

__all__ = [
    "Schema",
    "SchemaOutcome",
    "main",
    "real_matches",
    "relaxed_matches",
    "run_proposal",
    "schemas",
]

#: Saturation budget for a cost-change check on a real term.
_COST_ITERS = 5
_COST_NODES = 40_000

#: Firing budget (one law alone over a model).
_FIRE_ITERS = 4
_FIRE_NODES = 40_000


# ---------------------------------------------------------------------------
#  Schema library — honest, generic identities over the real vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Schema:
    """One generic equality ``lhs -> rhs`` over metavariable patterns.

    ``check`` is an optional side-condition hook over the resolved
    binding — the ``Rewrite.check`` contract.  A guarded schema mints
    only *conditionally*: every consumer (``_instance_from_match``,
    ``sweep_real``, ``verify_view_candidate``, e-graph firing) applies
    it, so a claim like ``reshape_transpose`` counts only the
    instances whose shapes prove the commutation aligned.
    """

    name: str
    lhs: Any
    rhs: Any
    note: str = ""
    family: str = ""
    check: Any = None

    def as_rule(self) -> Rewrite:
        """Return the schema as a fireable ``Rewrite``."""
        return Rewrite(self.name, self.lhs, self.rhs, check=self.check)


def schemas() -> list[Schema]:
    """Return the schema library, grouped by the shape it targets.

    The library is data — :data:`kit_discovery.lawdata.SHAPE_SCHEMAS`,
    one ``(name, lhs, rhs, note, family)`` spec row per schema.
    Every schema is a *true* algebraic identity over the real op
    vocabulary (elementwise arithmetic, the ``select``/``slice``
    naturality, layout).  Some are deliberately library duplicates
    (the ``*_factor`` controls) so the tool's duplicate detector is
    exercised, and some target shapes the census found frequent
    (``mul(select, select)``, ``add(mul, mul)``).  Schemas named in
    :data:`_SCHEMA_CHECKS` additionally carry their side condition —
    the minted claim is guarded, not refused outright, so the aligned
    corner can still verify.
    """
    out: list[Schema] = []
    for row in lawdata.SHAPE_SCHEMAS:
        s = _schema(row)
        check = _SCHEMA_CHECKS.get(s.name)
        out.append(replace(s, check=check) if check is not None else s)
    return out


def _schema(row: tuple) -> Schema:
    """Resolve one ``SHAPE_SCHEMAS`` row to a :class:`Schema`."""
    name, lhs, rhs, note, family = row
    return Schema(
        name,
        _spec_term(lhs),
        _spec_term(rhs),
        note=note,
        family=family,
    )


# ---------------------------------------------------------------------------
#  Side conditions — schemas whose equality needs a shape-level guard
# ---------------------------------------------------------------------------


def _concrete_shape(term: Any) -> tuple[int, ...] | None:
    """Return *term*'s inferred shape when fully concrete, else ``None``.

    Every dim must be a positive ``int`` — a ``None``/symbolic dim or
    a non-tuple answer means the guard cannot prove alignment, so it
    declines rather than fabricate one.
    """
    s = _shape_of(term)
    if not isinstance(s, tuple):
        return None
    if not all(
        isinstance(d, int) and not isinstance(d, bool) and d > 0
        for d in s
    ):
        return None
    return s


def _reshape_target(
    spec: Any, sa: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Resolve a bound ``shape`` attr to a concrete target tuple.

    Accepts a tuple/list of positive ints with at most one ``-1``
    (folded against *sa*'s numel, the torch convention) whose product
    equals *sa*'s — a valid reshape target.  Anything else (``None``
    dims, strings, multi-``-1``, numel mismatch) returns ``None``.
    """
    if isinstance(spec, bool):
        return None
    if isinstance(spec, int):
        spec = (spec,)
    if not isinstance(spec, (tuple, list)):
        return None
    dims: list[int] = []
    wild: list[int] = []
    for k, v in enumerate(spec):
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v == -1:
            wild.append(k)
        elif v <= 0:
            return None
        dims.append(v)
    if len(wild) > 1:
        return None
    if wild:
        known = math.prod(v for v in dims if v != -1)
        n = math.prod(sa)
        if known == 0 or n % known:
            return None
        dims[wild[0]] = n // known
    if math.prod(dims) != math.prod(sa):
        return None
    return tuple(dims)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides of *shape*."""
    w = [1] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        w[k] = w[k + 1] * shape[k + 1]
    return tuple(w)


def _perm_src(
    p: int,
    out_w: tuple[int, ...],
    in_w: tuple[int, ...],
    d: int,
    i: int,
) -> int:
    """Source flat index for output flat *p* under an axis swap.

    Decodes *p* under the *output* (swapped) shape's strides, swaps
    coordinates *d*/*i*, and re-encodes under the *input* strides —
    the flat permutation ``transpose`` induces on a tensor of the
    input shape.
    """
    rank = len(in_w)
    rem = p
    coords = [0] * rank
    for k in range(rank):
        coords[k] = rem // out_w[k]
        rem %= out_w[k]
    coords[d], coords[i] = coords[i], coords[d]
    src = 0
    for k in range(rank):
        src += coords[k] * in_w[k]
    return src


def _transpose_perm_data(
    shape: tuple[int, ...], d: int, i: int
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    """Precomputed ``(in_w, out_w, d, i)`` for one transpose."""
    r = len(shape)
    d %= r
    i %= r
    out = list(shape)
    out[d], out[i] = out[i], out[d]
    return _strides(shape), _strides(tuple(out)), d, i


#: Cap on the flat-index space a guard compares exhaustively — beyond
#: it the check declines (a conservative refuse, never a guess).
_PERM_CAP = 1 << 20


@lru_cache(maxsize=4096)
def _transpose_perms_equal(
    s1: tuple[int, ...],
    d1: int,
    i1: int,
    s2: tuple[int, ...],
    d2: int,
    i2: int,
) -> bool:
    """Return whether two axis swaps induce the same flat permutation.

    ``transpose(·, d1, i1)`` on shape *s1* and ``transpose(·, d2, i2)``
    on shape *s2* induce the SAME flat permutation of ``range(numel)``.

    This is the exact axis-alignment criterion for
    ``transpose(reshape(A, S)) = reshape(transpose(A))``: reshape
    preserves flat order, so the claim holds iff the digit-swap the
    outer transpose performs on the *reshaped* view equals the digit-
    swap the inner transpose performs on ``A`` — checked pointwise
    over every flat index (early exit on the first disagreement).
    """
    n1, n2 = math.prod(s1), math.prod(s2)
    if n1 != n2:
        return False
    if n1 > _PERM_CAP:
        return False
    iw1, ow1, a1, b1 = _transpose_perm_data(s1, d1, i1)
    iw2, ow2, a2, b2 = _transpose_perm_data(s2, d2, i2)
    for p in range(n1):
        if _perm_src(p, ow1, iw1, a1, b1) != _perm_src(
            p, ow2, iw2, a2, b2
        ):
            return False
    return True


def _check_reshape_transpose(bound: dict) -> bool:
    """Guard ``transpose(reshape(A, S), D, I) = reshape(transpose(A, D, I), S)``.

    Bare axis metavariables make the schema's commutation *conditional*:
    the reshape's split/merge must leave the transposed axes aligned.
    The guard decides that per binding — both shapes must be concrete,
    the reshape target must resolve, each transpose's axes must be
    valid on its own tensor, the LHS transpose must preserve the
    output shape (``st[d] == st[i]``), and the two induced flat
    permutations must agree on every index.  Anything unresolvable —
    unknown leaf shapes, ``None`` dims, tensors past the comparison
    cap — declines: the claim mints only where alignment is proven.
    """
    sa = _concrete_shape(bound.get("A"))
    st = (
        _reshape_target(bound.get("$attr:S"), sa)
        if sa is not None
        else None
    )
    d, i = bound.get("$attr:D"), bound.get("$attr:I")
    if (
        sa is None
        or st is None
        or not sa
        or not st
        or isinstance(d, bool)
        or isinstance(i, bool)
        or not isinstance(d, int)
        or not isinstance(i, int)
    ):
        return False
    rt, rs = len(st), len(sa)
    if not (-rt <= d < rt and -rt <= i < rt):
        return False
    if not (-rs <= d < rs and -rs <= i < rs):
        return False
    dl, il = d % rt, i % rt
    dr, ir = d % rs, i % rs
    # The LHS output shape is ``swap(st, dl, il)``; the RHS output
    # shape is ``st`` — unequal extents make the instance ill-formed.
    if dl != il and st[dl] != st[il]:
        return False
    return _transpose_perms_equal(st, dl, il, sa, dr, ir)


#: Schema name → procedural ``check`` hook (the ``Rewrite.check``
#: contract over the resolved ``{metavar: term, "$attr:NAME": value}``
#: binding).  Attached by :func:`schemas` so the minted proposal —
#: and every oracle/gauntlet path that consumes it — honours the
#: side condition instead of evaluating misaligned instances as
#: counterexamples.
_SCHEMA_CHECKS: dict[str, Any] = {
    "reshape_transpose": _check_reshape_transpose,
}


# ---------------------------------------------------------------------------
#  Match counts — does the shape (and its equality) appear?
# ---------------------------------------------------------------------------


def _relax(lhs: Any) -> Any:
    """Rename each repeated metavariable to a fresh name (wildcards)."""
    counts: dict[str, int] = {}

    def walk(t: Any) -> Any:
        if isinstance(t, Op):
            return Op.make(
                t.op,
                *(walk(a) for a in t.args),
                **dict(t.attrs),
            )
        if isinstance(t, str):
            counts[t] = counts.get(t, 0) + 1
            return t if counts[t] == 1 else f"{t}__{counts[t]}"
        return t

    return walk(lhs)


def real_matches(terms: list[Any], schema: Schema) -> list[Any]:
    """Return every real subterm the schema's LHS matches (equalities on).

    ``_term_match`` enforces that a repeated metavariable binds a
    structurally equal subterm, so this is the *applicable* count, not
    the shape count.  Subterms are deduped by identity (matching the
    census's ``_iter_subterms``), so a shared node is counted once, not
    once per path.
    """
    out: list[Any] = []
    for term in terms:
        for sub in _iter_subterms(term):
            if not isinstance(sub, Op):
                continue
            if _term_match(schema.lhs, sub) is not None:
                out.append(sub)
    return out


def relaxed_matches(terms: list[Any], schema: Schema) -> int:
    """Count real subterms matching the LHS with metavars relaxed.

    Repeated *argument* metavariables become wildcards, so this counts
    the pure shape (e.g. ``add(mul, mul)``); attribute metavariables are
    left shared, so a shape whose precondition is on attributes (the
    ``select`` dim/index) still enforces it.  Deduped by identity.
    """
    pat = _relax(schema.lhs)
    n = 0
    for term in terms:
        for sub in _iter_subterms(term):
            if (
                isinstance(sub, Op)
                and _term_match(pat, sub) is not None
            ):
                n += 1
    return n


# ---------------------------------------------------------------------------
#  Per-schema outcome
# ---------------------------------------------------------------------------


@dataclass
class SchemaOutcome:
    """One schema plus every measured verdict."""

    schema: Schema
    relaxed: int = 0
    matches: int = 0
    example: str = ""
    num_true: bool | None = None
    derivable: bool = False
    relation: str = "new"
    base_cost: float = 0.0
    cand_cost: float = 0.0
    model_fires: int = 0
    fire_cases: tuple[str, ...] = ()
    fire_changed: int = 0
    fire_paid: int = 0

    @property
    def new(self) -> bool:
        """True iff not a duplicate or inverse of a library law."""
        return self.relation == "new"

    @property
    def truth(self) -> bool:
        """True iff derivable, or numerically true on real shapes."""
        return self.derivable or self.num_true is True

    @property
    def useful(self) -> bool:
        """True iff a *true* rule strictly lowered a real term's cost."""
        return self.truth and self.cand_cost < self.base_cost


# ---------------------------------------------------------------------------
#  Truth / derivability / newness
# ---------------------------------------------------------------------------


def _instantiate(schema: Schema, sub: Any) -> Any | None:
    """Return the RHS instance of *schema* on the matched subterm *sub*.

    The schema's ``check`` side condition (when present) vetoes the
    match exactly as a firing would — the numeric oracle sees only
    instances whose shapes prove the claim, never a misaligned
    spelling counted as a counterexample.
    """
    rule = schema.as_rule()
    subst = _term_match(schema.lhs, sub)
    if subst is None:
        return None
    if rule.check is not None:
        try:
            if not rule.check(subst):
                return None
        except Exception:
            return None
    return _term_instantiate(schema.rhs, subst)


# ---------------------------------------------------------------------------
#  Cost delta + firing on real graphs
# ---------------------------------------------------------------------------


def _cost_delta(
    schema: Schema, matches: list[Any], cost_fn: Any
) -> tuple:
    """Return ``(base, cand)`` costs over the schema's matched terms."""
    rule = schema.as_rule()
    base = float("inf")
    with_rule = float("inf")
    for term in matches[:8]:
        base = min(base, _sat_cost(term, list(ALL_RULES), cost_fn))
        with_rule = min(
            with_rule,
            _sat_cost(term, [*ALL_RULES, rule], cost_fn),
        )
    return base, with_rule


def _fires_on_models(
    schema: Schema, models: list[Any], cost_fn: Any
) -> tuple:
    """Run *schema* alone over every model; return firing aggregates."""
    rule = schema.as_rule()
    fires = 0
    cases: list[str] = []
    changed = 0
    paid = 0
    for c in models:
        eg = EGraph()
        root = eg.add_term(c.term)
        eg.run(
            [rule],
            root,
            max_iterations=_FIRE_ITERS,
            max_nodes=_FIRE_NODES,
        )
        n = eg.rule_fires.get(schema.name, 0)
        if not n:
            continue
        fires += n
        cases.append(c.name)
        best = eg.extract_best(root, cost_fn)
        if best is None or best == c.term:
            continue
        changed += 1
        if dag_cost(best, cost_fn) < dag_cost(c.term, cost_fn):
            paid += 1
    return fires, tuple(cases), changed, paid


_SINK: Any = None


def _sink() -> Any:
    """Return a shared ``TorchSink`` (built once)."""
    global _SINK
    if _SINK is None:
        from kit_torch.adapters import TorchSink

        _SINK = TorchSink()
    return _SINK


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def run_proposal() -> dict:
    """Measure every schema over the real corpus; return the result."""
    bench, _be = _bench_cases()
    models, _me = model_cases()
    real = [c.term for c in [*bench, *models]]
    cost_fn = _cost_fn(_sink())
    lib = _library_keys()

    outcomes: list[SchemaOutcome] = []
    for schema in schemas():
        out = SchemaOutcome(schema=schema)
        out.relaxed = relaxed_matches(real, schema)
        matches = real_matches(real, schema)
        out.matches = len(matches)
        out.relation = _relation(schema.lhs, schema.rhs, lib)
        if matches:
            from mlopt_core.ir import op_repr

            out.example = op_repr(matches[0])
            # First match whose binding survives the schema's guard —
            # a guarded schema's truth is measured on the aligned
            # corner, not on a vetoed misaligned spelling.
            inst = next(
                (
                    (m, rhs)
                    for m in matches
                    if (rhs := _instantiate(schema, m)) is not None
                ),
                None,
            )
            if inst is not None:
                out.num_true = _numeric_true(inst[0], inst[1])
                out.derivable = verify_law(
                    inst[0], inst[1], ALL_RULES
                ).derivable
            out.base_cost, out.cand_cost = _cost_delta(
                schema, matches, cost_fn
            )
        (
            out.model_fires,
            out.fire_cases,
            out.fire_changed,
            out.fire_paid,
        ) = _fires_on_models(schema, models, cost_fn)
        outcomes.append(out)

    return {
        "n_bench": len(bench),
        "n_models": len(models),
        "outcomes": outcomes,
        "baseline": _baseline(models),
        "reach": _reach(outcomes, models, cost_fn),
    }


def _reach(
    outcomes: list[SchemaOutcome],
    models: list[Any],
    cost_fn: Any,
) -> list[dict]:
    """End-to-end reach for every schema that fires on a real model.

    Saturates each model under ``ALL_RULES`` vs
    ``ALL_RULES + {schema}`` and reports the extracted-cost delta and
    whether the certificate still replays — the same measurement the
    impact retro used, so a "pays" claim is end-to-end, not just on a
    matched subterm.
    """
    sink = _sink()
    rows: list[dict] = []
    for o in outcomes:
        if not o.model_fires:
            continue
        for c in models:
            r = reach_row(c, [o.schema.as_rule()], sink, cost_fn)
            if not r["new_fires"] and not r["changed"]:
                continue
            rows.append(
                {
                    "schema": o.schema.name,
                    "model": r["model"],
                    "base_cost": r["base_cost"],
                    "add_cost": r["add_cost"],
                    "changed": r["changed"],
                    "fires": sum(r["new_fires"].values()),
                    "cert": r["add_cert"],
                }
            )
    return rows


def _baseline(models: list[Any]) -> dict:
    """Measure the 11 earlier laws for comparison (the impact baseline)."""
    rules = new_laws()
    fired: dict[str, int] = {}
    for c in models:
        eg = EGraph()
        root = eg.add_term(c.term)
        eg.run(
            rules,
            root,
            max_iterations=_FIRE_ITERS,
            max_nodes=_FIRE_NODES,
        )
        for r in rules:
            n = eg.rule_fires.get(r.name, 0)
            if n:
                fired[r.name] = fired.get(r.name, 0) + n
    return {"laws": [r.name for r in rules], "fires": fired}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _match_table(outs: list[SchemaOutcome]) -> str:
    """Render the shape-match / truth / usefulness table."""
    head = (
        f"{'schema':<20} {'relax':>6} {'match':>6} {'true':>5} "
        f"{'deriv':>5} {'rel':>10} {'cost':>18} {'use':>4}"
    )
    lines = [head, "-" * len(head)]
    for o in outs:
        true = {True: "yes", False: "no", None: "-"}[o.num_true]
        cost = (
            f"{o.base_cost:.3g}->{o.cand_cost:.3g}"
            if o.matches
            else "-"
        )
        lines.append(
            f"{o.schema.name:<20} {o.relaxed:>6} {o.matches:>6} "
            f"{true:>5} {o.derivable!s:>5} {o.relation:>10} "
            f"{cost:>18} {'yes' if o.useful else 'no':>4}"
        )
    return "\n".join(lines)


def _fire_table(outs: list[SchemaOutcome]) -> str:
    """Render the per-schema firing table on the real models."""
    head = (
        f"{'schema':<20} {'fires':>6} {'changed':>8} {'paid':>5}  cases"
    )
    lines = [head, "-" * len(head)]
    for o in outs:
        cases = ",".join(o.fire_cases[:3])
        if len(o.fire_cases) > 3:
            cases += "…"
        lines.append(
            f"{o.schema.name:<20} {o.model_fires:>6} "
            f"{o.fire_changed:>8} {o.fire_paid:>5}  {cases}"
        )
    return "\n".join(lines)


def _reach_table(rows: list[dict]) -> str:
    """Render the end-to-end reach rows."""
    if not rows:
        return "  (no schema changed a real model end-to-end)"
    head = (
        f"{'schema':<18} {'model':<18} {'cost':>22} "
        f"{'fires':>6} {'cert':>5}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        cost = f"{r['base_cost']:.3g}->{r['add_cost']:.3g}"
        lines.append(
            f"{r['schema']:<18} {r['model']:<18} {cost:>22} "
            f"{r['fires']:>6} {r['cert']:>5}"
        )
    return "\n".join(lines)


def _verdict(result: dict) -> None:
    """Print the plain verdict the retro records."""
    outs: list[SchemaOutcome] = result["outcomes"]
    real = [o for o in outs if o.matches]
    true_new = [o for o in real if o.truth and o.new]
    useful = [o for o in outs if o.useful and o.new]
    fires = [o for o in outs if o.model_fires]
    pays = [o for o in outs if o.model_fires and o.fire_paid]
    true_pays = [o for o in pays if o.truth and o.new]
    false_pays = [o for o in pays if not (o.truth and o.new)]
    print("== verdict ==")  # stdout-compat
    print(  # stdout-compat
        f"  schemas: {len(outs)}; with a real (equality-enforced) "
        f"match: {len(real)}"
    )
    print(f"  true & new among those: {len(true_new)}")  # stdout-compat
    print(  # stdout-compat
        f"  useful (cost-lowering on a real term): {len(useful)}"
    )
    print(f"  firing on a real model: {len(fires)}")  # stdout-compat
    print(f"  firing AND cost-lowering: {len(pays)}")  # stdout-compat
    print(f"    of which true & new: {len(true_pays)}")  # stdout-compat
    print(  # stdout-compat
        f"    of which false/duplicate (cost-only): {len(false_pays)}"
    )
    base = result["baseline"]
    print(  # stdout-compat
        f"  baseline (11 earlier laws): fires on models = "
        f"{sum(base['fires'].values())}"
    )
    if true_pays:
        print("  paying & true schemas:")  # stdout-compat
        for o in true_pays:
            print(  # stdout-compat
                f"    - {o.schema.name}: fires={o.model_fires}, "
                f"paid={o.fire_paid}, cases={o.fire_cases}"
            )
    if false_pays:
        print(  # stdout-compat
            "  cost-lowering but NOT true (oracle rejects — the "
            "reason a cost-only proposer is unsafe):"
        )
        for o in false_pays:
            print(  # stdout-compat
                f"    - {o.schema.name}: num_true={o.num_true}, "
                f"relation={o.relation}"
            )


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable result."""
    outs: list[SchemaOutcome] = result["outcomes"]
    payload = {
        "n_bench": result["n_bench"],
        "n_models": result["n_models"],
        "baseline": result["baseline"],
        "reach": result["reach"],
        "outcomes": [
            {
                "schema": o.schema.name,
                "family": o.schema.family,
                "note": o.schema.note,
                "relaxed": o.relaxed,
                "matches": o.matches,
                "example": o.example,
                "num_true": o.num_true,
                "derivable": o.derivable,
                "relation": o.relation,
                "base_cost": o.base_cost,
                "cand_cost": o.cand_cost,
                "useful": o.useful,
                "model_fires": o.model_fires,
                "fire_cases": list(o.fire_cases),
                "fire_changed": o.fire_changed,
                "fire_paid": o.fire_paid,
            }
            for o in outs
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the shape-aware proposal and print (or dump) the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    result = run_proposal()
    outs: list[SchemaOutcome] = result["outcomes"]

    print(  # stdout-compat
        "== law_shape_proposal — laws aimed at real shapes =="
    )
    print(  # stdout-compat
        f"   corpus: {result['n_bench']} bench + "
        f"{result['n_models']} models; {len(outs)} schemas"
    )
    print()  # stdout-compat
    print(  # stdout-compat
        "-- match / truth / usefulness (relax = shape only) --"
    )
    print(_match_table(outs))  # stdout-compat
    print()  # stdout-compat
    print(  # stdout-compat
        "-- firing on real models (each schema run alone) --"
    )
    print(_fire_table(outs))  # stdout-compat
    print()  # stdout-compat
    print(  # stdout-compat
        "-- examples (first real match per applicable schema) --"
    )
    for o in outs:
        if o.example:
            print(  # stdout-compat
                f"  {o.schema.name:<20} {o.example[:90]}"
            )
    print()  # stdout-compat
    print(  # stdout-compat
        "-- end-to-end reach (ALL_RULES vs ALL_RULES + schema) --"
    )
    print(_reach_table(result["reach"]))  # stdout-compat
    print()  # stdout-compat
    _verdict(result)

    if args.json:
        _dump_json(args.json, result)
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
