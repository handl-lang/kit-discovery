"""Tests for ``kit_discovery.law_evo`` — the evolutionary law-bank pilot.

Torch-free: every expensive seam is injected — ``cases_fn`` builds
tiny ``TermCase``s over hand-made ``Op`` terms, ``pool_fn`` returns
hand-built ``pipeline.Proposal``s, and ``measure_fn`` returns
stub verdict objects.  What is *not* stubbed is the machinery under
test: ``armed_keys`` (real ``_term_match`` over real terms),
``fitness`` (the coverage/pay union), ``evolve`` (the GA), the
ledger's resume semantics, and ``run_experiment`` end to end.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from mlopt_core.ir import Op, TensorType, Var

from kit_discovery import law_evo as le
from kit_discovery.impact import TermCase
from kit_discovery.pipeline import Proposal


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _case(name: str, term: Op) -> TermCase:
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=(),
        feed=(),
        param_vals={},
    )


def _terms() -> list[Op]:
    x, y, w = _v("x", 4), _v("y", 4), _v("w", 4)
    return [
        Op.make("add", Op.make("mul", x, w), y),
        Op.make("add", Op.make("mul", y, w), x),
        Op.make("mul", Op.make("add", x, y), w),
    ]


def _cases() -> list[TermCase]:
    return [_case(f"t{i}", t) for i, t in enumerate(_terms())]


def _proposal(name: str, lhs: Op, rhs: Op) -> Proposal:
    return Proposal(name=name, lhs=lhs, rhs=rhs, family="test")


def _pool() -> list[Proposal]:
    return [
        _proposal(
            "add_mul",
            Op.make("add", "M0", "M1"),
            Op.make("mul", "M0", "M1"),
        ),
        _proposal(
            "mul_add",
            Op.make("mul", "M0", "M1"),
            Op.make("add", "M0", "M1"),
        ),
        _proposal(
            "deep",
            Op.make("add", Op.make("mul", "M0", "M1"), "M2"),
            Op.make("mul", "M0", "M1"),
        ),
    ]


def _verdict(shippable: bool, paid: int = 0, **kw) -> SimpleNamespace:
    base = dict(
        shippable=shippable,
        paid=paid,
        fires=paid,
        cost_drop=0.1 if paid else 0.0,
        num_true=True if shippable else None,
        relation="new",
        matches=1,
        no_ship_reason="" if shippable else "no firing on a real model",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _measure_fn(p: Proposal, cases, census_op) -> SimpleNamespace:
    # Deterministic per-name verdicts: two shippable, one refused.
    table = {
        "add_mul": _verdict(True, paid=2),
        "mul_add": _verdict(True, paid=1),
        "deep": _verdict(False),
    }
    return table[p.name]


def _run(tmp_path, **kw) -> dict:
    args = dict(
        ledger_path=tmp_path / "evo.jsonl",
        generations=3,
        pop_size=4,
        set_size=2,
        pool_cap=8,
        seed=0,
        cases_fn=_cases,
        pool_fn=lambda census_op, terms: _pool(),
        measure_fn=_measure_fn,
    )
    args.update(kw)
    return le.run_experiment(**args)


# ---------------------------------------------------------------------------
#  Units
# ---------------------------------------------------------------------------


def test_armed_keys_counts_applicable_sites():
    cases = _cases()
    terms = [c.term for c in cases]
    p = _proposal(
        "add_mul",
        Op.make("add", "M0", "M1"),
        Op.make("mul", "M0", "M1"),
    )
    keys = le.armed_keys(p, terms)
    # Every 'add' Op subterm across the three terms arms the LHS.
    n_adds = sum(
        1
        for t in terms
        for s in le._iter_subterms(t)
        if isinstance(s, Op) and s.op == "add"
    )
    assert len(keys) == n_adds
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keys)


def test_armed_keys_respects_repeated_metavars():
    cases = _cases()
    terms = [c.term for c in cases]
    # add(M0, M0) only matches adds whose two args are equal — none here.
    p = _proposal("idem", Op.make("add", "M0", "M0"), "M0")
    assert le.armed_keys(p, terms) == frozenset()


def test_fitness_unions_sites_and_averages_pay():
    a = le.LawRecord(
        name="a",
        family="t",
        rank=0,
        shippable=True,
        paid=2,
        fires=2,
        cost_drop=0.1,
        num_true=True,
        relation="new",
        matches=1,
        no_ship_reason="",
        armed=frozenset({(0, 0), (0, 1)}),
    )
    b = le.LawRecord(
        name="b",
        family="t",
        rank=1,
        shippable=True,
        paid=1,
        fires=1,
        cost_drop=0.1,
        num_true=True,
        relation="new",
        matches=1,
        no_ship_reason="",
        armed=frozenset({(0, 1), (1, 0)}),
    )
    f, cov, pay = le.fitness(
        [a, b], n_sites=4, n_models=2, pay_weight=0.5
    )
    # coverage = |{(0,0),(0,1),(1,0)}| / 4 = 0.75
    assert cov == 0.75
    # pay_rate = (2+1) / (2 members * 2 probes) = 0.75
    assert pay == 0.75
    assert f == 0.75 + 0.5 * 0.75


def test_fitness_empty_set_is_zero():
    assert le.fitness([], n_sites=10, n_models=3) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
#  GA
# ---------------------------------------------------------------------------


def _records() -> list[le.LawRecord]:
    # Four survivors with overlapping armed sets + paid counts.
    specs = [
        ("s0", frozenset({(0, 0)}), 2, 0),
        ("s1", frozenset({(0, 1)}), 1, 1),
        ("s2", frozenset({(1, 0), (1, 1)}), 0, 2),
        (
            "dead",
            frozenset({(2, 0)}),
            5,
            3,
        ),  # shippable=False — excluded
    ]
    out = []
    for _i, (name, armed, paid, rank) in enumerate(specs):
        out.append(
            le.LawRecord(
                name=name,
                family="t",
                rank=rank,
                shippable=name != "dead",
                paid=paid,
                fires=paid,
                cost_drop=0.0,
                num_true=True,
                relation="new",
                matches=1,
                no_ship_reason="",
                armed=armed,
            )
        )
    return out


def test_evolve_gauntlet_filters_and_selects(tmp_path):
    ledger = le.EvoLedger(tmp_path / "e.jsonl")
    res = le.evolve(
        _records(),
        generations=3,
        pop_size=4,
        set_size=2,
        seed=0,
        n_probes=3,
        ledger=ledger,
    )
    assert res["n_shippable"] == 3  # 'dead' never enters the gene pool
    assert res["n_sites"] == 4
    # best set is non-empty, all members shippable, fitness > baseline-able
    assert res["best"]["members"]
    assert set(res["best"]["members"]) <= {"s0", "s1", "s2"}
    assert res["best"]["fitness"] > 0
    # baseline = first set_size survivors in rank order
    assert res["baseline"]["members"] == ["s0", "s1"]
    # the GA must do at least as well as the baseline arm
    assert res["best"]["fitness"] >= res["baseline"]["fitness"] - 1e-9
    # ledger received gen + eval + (no measure — evolve doesn't measure)
    kinds = [
        json.loads(line)["kind"]
        for line in (tmp_path / "e.jsonl").read_text().splitlines()
    ]
    assert kinds.count("gen") == 3
    assert "eval" in kinds


def test_evolve_is_deterministic(tmp_path):
    r1 = le.evolve(
        _records(),
        generations=2,
        pop_size=4,
        set_size=2,
        seed=7,
        n_probes=3,
    )
    r2 = le.evolve(
        _records(),
        generations=2,
        pop_size=4,
        set_size=2,
        seed=7,
        n_probes=3,
    )
    assert r1["best"] == r2["best"]
    assert [g["best"] for g in r1["history"]] == [
        g["best"] for g in r2["history"]
    ]


def test_evolve_handles_empty_survivor_pool(tmp_path):
    dead = [
        le.LawRecord(
            name="d",
            family="t",
            rank=0,
            shippable=False,
            paid=0,
            fires=0,
            cost_drop=0.0,
            num_true=False,
            relation="dup",
            matches=0,
            no_ship_reason="false",
            armed=frozenset(),
        )
    ]
    res = le.evolve(
        dead, generations=2, pop_size=4, set_size=2, seed=0, n_probes=1
    )
    assert res["n_shippable"] == 0
    assert res["best"]["members"] == []
    assert res["best"]["fitness"] == 0.0


# ---------------------------------------------------------------------------
#  Ledger + end-to-end
# ---------------------------------------------------------------------------


def test_ledger_resume_skips_measure(tmp_path):
    path = tmp_path / "evo.jsonl"
    calls = []

    def counting(p, cases, census_op):
        calls.append(p.name)
        return _measure_fn(p, cases, census_op)

    _run(tmp_path, measure_fn=counting)
    first = list(calls)
    assert first  # the first pass measured everything

    # Second run on the same ledger: measure_fn must not be called again.
    calls.clear()
    res2 = _run(tmp_path, measure_fn=counting)
    assert calls == []
    assert res2["n_measured"] == len(_pool())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(1 for r in rows if r.get("kind") == "measure") == len(
        _pool()
    )


def test_run_experiment_end_to_end(tmp_path):
    res = _run(tmp_path)
    assert res["kind"] == "summary"
    assert res["n_pool"] == 3
    assert res["n_measured"] == 3
    assert res["n_shippable"] == 2
    assert res["gauntlet_pass_rate"] == round(2 / 3, 4)
    assert res["candidates_per_hour"] > 0
    # the refused candidate is named in a measure row
    rows = [
        json.loads(line)
        for line in (tmp_path / "evo.jsonl").read_text().splitlines()
    ]
    deep = next(r for r in rows if r.get("name") == "deep")
    assert deep["shippable"] is False
    assert res["best"]["members"]


def test_census_op_for_and_pool_smoke(tmp_path):
    # The real (un-injected) census path over the tiny corpus.
    cases = _cases()
    census_op = le.census_op_for(cases)
    assert census_op  # the terms carry real op tuples
    terms = [c.term for c in cases]
    pool = le.gen_pool(census_op, terms, cap=4)
    assert isinstance(pool, list)
