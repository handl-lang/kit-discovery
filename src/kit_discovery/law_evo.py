"""Experiment 2 — evolutionary selection over the law bank.

The cheap-reward RL scaling plan's second experiment: instead of the
overnight mint's *generate → gauntlet → ledger* linear pass, run an
actual evolutionary loop over candidate law-**sets** —

    pool → cheap-gauntlet filter → fitness → tournament select → N gens

where a candidate is a :class:`kit_discovery.pipeline.Proposal` (the
same object ``pipeline.propose`` mints for the ``zoo-real`` arm), the
cheap gauntlet is ``pipeline.measure``'s verdict bundle on a small
probe suite (``Evidence.shippable`` — truth + novelty + typed fires +
paid + closure, the same gates ``run_pipeline`` ranks by, minus the
8-stage stored-object gauntlet and the bench referees), and fitness is

    fitness(S) = armed-coverage(S) + ``pay_weight`` · pay-rate(S)

*armed-coverage* is the fraction of probe-corpus subterm sites any
member's LHS pattern matches (the ``real_matches`` applicable-site
count, unioned across the set — the coverage-gain half of the
hypothesis); *pay-rate* is the mean per-capita ``Evidence.paid``
count over the probe models (the measured-pay half).  Both are
computed from per-law measurements cached in the JSONL ledger, so a
set evaluation costs microseconds once its members are measured —
the expensive step (``pipeline.measure``: numeric oracle,
derivability, firing, reach) runs **once per candidate**, exactly
like the mint path, and the evolution only recombines survivors.

Resumability: every measured candidate appends a ``measure`` JSONL
row; on restart the pool is rebuilt deterministically and
already-measured names skip ``pipeline.measure`` entirely — the
expensive half of the run.  Evaluated member-sets are appended as
``eval`` rows for audit; fitness is cheap enough to recompute, so
the GA replays deterministically from the same seed rather than
needing an eval cache.  The loop picks up where the ledger left off.

Honesty contract: this is a *pilot driver*, not an admission path.
Nothing here writes to ``mlopt_core.laws`` / ``DEFAULT`` / the
evidence store's AUTO_TIER — the "bank" under evolution is the
in-memory survivor set plus the ledger.  The baseline arm replays
the mint order (first ``set_size`` survivors in census rank) so the
report can compare admitted-quality yield per hour honestly; a
negative result is a valid result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlopt_core.egraph.terms import _term_match
from mlopt_core.ir import Op
from mlopt_core.laws import ALL_RULES

from kit_discovery import census as _census
from kit_discovery import pipeline as _pipeline
from kit_discovery import proposal as _lp
from kit_discovery.impact import (
    TermCase,
    _cost_fn,
    _iter_subterms,
    model_cases,
    synthetic_cases,
)
from kit_discovery.shape_proposal import Schema, _sink, relaxed_matches

__all__ = [
    "EvoLedger",
    "LawRecord",
    "armed_keys",
    "build_cases",
    "census_op_for",
    "evolve",
    "fitness",
    "gen_pool",
    "measure_pool",
    "run_experiment",
]


# ---------------------------------------------------------------------------
#  Records
# ---------------------------------------------------------------------------


@dataclass
class LawRecord:
    """One measured candidate: the verdict scalars fitness reads.

    Deliberately *not* the full :class:`pipeline.Evidence` — the GA
    needs ``shippable`` (the cheap-gauntlet verdict), ``paid`` (the
    probe pay count) and the precomputed ``armed`` site set; keeping
    the record flat makes ledger resume a pure-data rebuild.
    """

    name: str
    family: str
    rank: int  # position in the census-ranked proposal pool
    shippable: bool
    paid: int
    fires: int
    cost_drop: float
    num_true: bool | None
    relation: str
    matches: int
    no_ship_reason: str
    armed: frozenset = field(default_factory=frozenset)

    @classmethod
    def from_evidence(
        cls,
        proposal: Any,
        ev: Any,
        rank: int,
        armed: frozenset,
    ) -> LawRecord:
        """Flatten a ``pipeline.Evidence`` into the ledger record."""
        return cls(
            name=proposal.name,
            family=proposal.family,
            rank=rank,
            shippable=bool(ev.shippable),
            paid=int(ev.paid),
            fires=int(ev.fires),
            cost_drop=float(ev.cost_drop),
            num_true=ev.num_true,
            relation=str(ev.relation),
            matches=int(ev.matches),
            no_ship_reason=str(ev.no_ship_reason),
            armed=armed,
        )

    def to_row(self) -> dict[str, Any]:
        """Return the JSONL ``measure`` row (scalar fields only)."""
        return {
            "kind": "measure",
            "name": self.name,
            "family": self.family,
            "rank": self.rank,
            "shippable": self.shippable,
            "paid": self.paid,
            "fires": self.fires,
            "cost_drop": self.cost_drop,
            "num_true": self.num_true,
            "relation": self.relation,
            "matches": self.matches,
            "no_ship_reason": self.no_ship_reason,
        }

    @classmethod
    def from_row(
        cls, row: dict[str, Any], rank: int, armed: frozenset
    ) -> LawRecord:
        """Rebuild a record from its ledger row (resume path)."""
        return cls(
            name=row["name"],
            family=row.get("family", ""),
            rank=rank,
            shippable=bool(row["shippable"]),
            paid=int(row["paid"]),
            fires=int(row["fires"]),
            cost_drop=float(row.get("cost_drop", 0.0)),
            num_true=row.get("num_true"),
            relation=row.get("relation", ""),
            matches=int(row.get("matches", 0)),
            no_ship_reason=row.get("no_ship_reason", ""),
            armed=armed,
        )


# ---------------------------------------------------------------------------
#  Corpus + pool
# ---------------------------------------------------------------------------


def build_cases(
    n_models: int = 6,
    n_synth: int = 4,
    *,
    model_fn: Callable | None = None,
    synth_fn: Callable | None = None,
) -> list[TermCase]:
    """Return the probe suite: a slice of exported models + synthetic witnesses.

    ``model_fn``/``synth_fn`` are the injection seams (tests pass
    fakes; the CLI default is ``impact.model_cases`` /
    ``impact.synthetic_cases``).  Synthetic witnesses come *first*:
    ``measure_pool``'s ``measure_n`` slice reads ``cases[:n]`` for
    the expensive fire/reach leg, and the designed witnesses are the
    cheap sites where a candidate can actually fire and pay — a slice
    of only big model graphs starves the pay signal (measured: 0/36
    shippable on a models-only slice vs the witnesses' paying sites).
    Coverage/matching always reads every case regardless of order.
    """
    mfn = model_cases if model_fn is None else model_fn
    sfn = synthetic_cases if synth_fn is None else synth_fn
    models, _errs = mfn()
    syn = sfn()
    return [*syn[:n_synth], *models[:n_models]]


def census_op_for(cases: list[TermCase]) -> dict:
    """Return the ``{(op, child-ops): count}`` census ``pipeline.propose`` reads.

    Same key shape ``census.op_tuple_census`` produces and
    ``pipeline._lhs_tuple`` looks up — computed over the probe suite
    only (the full ``run_census`` corpus needs ``bench.suites``, which
    lives outside this package's dep closure).
    """
    terms = [
        _census.CorpusTerm(c.source, c.name, c.term) for c in cases
    ]
    counts, _terms_of = _census.op_tuple_census(terms)
    return {
        (op, tuple(children)): n for (op, children), n in counts.items()
    }


def _lhs_tuple(term: Any) -> tuple | None:
    """``pipeline._lhs_tuple`` — the census op-tuple key of *term*."""
    if not isinstance(term, Op):
        return None
    return (term.op, tuple(_census._op_of(a) for a in term.args))


def gen_pool(
    census_op: dict,
    terms: list[Any],
    *,
    # ``"hand"`` not ``"derived"``: ``vocab.derive_vocabulary`` reads
    # ``impact._bench_cases`` → ``bench.suites``, which lives in the
    # mlopt repo outside this package's dep closure.  The hand
    # alphabet is the same one the pre-derived pipeline used.
    vocab: str = "hand",
    cap: int = 32,
    require_bind: bool = True,
) -> list[Any]:
    """Rank ``pipeline.propose``'s candidates and return the top *cap*.

    The rank is the cheap pre-gauntlet ordering: census op-tuple sites
    plus relaxed LHS matches over the probe terms (the same signals
    ``measure`` will confirm expensively).  Ties break on the alpha
    key so the pool — and therefore every downstream ledger row — is
    deterministic.
    """
    props = _pipeline.propose(
        census_op, terms, vocab, require_bind=require_bind
    )
    # Phase 1 (cheap): census op-tuple sites only — a dict lookup per
    # proposal.  Phase 2 (matching): relaxed LHS matches, but only on
    # the site-positive head of the ranking so a huge pool cannot
    # turn the pre-gauntlet sort into the run's bottleneck.
    keyed: list[tuple[int, str, Any]] = []
    for p in props:
        sites = census_op.get(_lhs_tuple(p.lhs), 0)
        key = repr(_lp._key(p.lhs, p.rhs))
        keyed.append((sites, key, p))
    keyed.sort(key=lambda t: (-t[0], t[1]))
    head = keyed[: max(cap * 8, cap)]
    scored: list[tuple[float, str, Any]] = []
    for sites, key, p in head:
        if sites <= 0:
            continue
        rel = relaxed_matches(terms, Schema(p.name, p.lhs, p.rhs))
        scored.append((float(sites + rel), key, p))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [p for _s, _k, p in scored[:cap]]


# ---------------------------------------------------------------------------
#  Cheap gauntlet + armed coverage
# ---------------------------------------------------------------------------


def armed_keys(proposal: Any, terms: list[Any]) -> frozenset:
    """Return the ``(term_index, subterm_index)`` sites *proposal*'s LHS arms.

    The applicable-match census — ``shape_proposal.real_matches``
    with indices kept — over the probe terms.  ``_iter_subterms``
    dedups by identity in a deterministic order, so the key set is
    stable across runs and across ledger resume.
    """
    out: set[tuple[int, int]] = set()
    lhs = proposal.lhs
    for ti, term in enumerate(terms):
        for si, sub in enumerate(_iter_subterms(term)):
            if not isinstance(sub, Op):
                continue
            if _term_match(lhs, sub) is not None:
                out.add((ti, si))
    return frozenset(out)


def measure_pool(
    proposals: list[Any],
    cases: list[TermCase],
    census_op: dict,
    *,
    view_oracle: bool = True,
    measure_n: int | None = None,
    ledger: EvoLedger | None = None,
    measure_fn: Callable | None = None,
    verbose: bool = False,
) -> tuple[list[LawRecord], int]:
    """Run the cheap gauntlet over *proposals*; return ``(records, n_probes)``.

    ``measure_fn`` is the injection seam: ``None`` runs the real
    ``pipeline.measure`` (numeric + derivability + firing + reach on
    the probe suite); tests pass a stub.  When *ledger* is given,
    candidates whose ``measure`` row already exists are rebuilt from
    it (the resume path) — only the ``armed`` set is recomputed,
    which is pure matching, no e-graph.

    ``measure_n`` bounds the *firing/reach* probe subset: matching
    (``real_terms``, ``armed_keys``) always reads every case, but
    ``pipeline.measure``'s ``models`` argument — the per-case
    e-graph fire + ALL_RULES reach saturation, the run's expensive
    leg — is sliced to ``cases[:measure_n]``.  ``None`` measures
    every case.  The returned ``n_probes`` is the size of that
    subset — the denominator :func:`evolve`'s pay-rate uses.
    """
    terms = [c.term for c in cases]
    probe_cases = cases if measure_n is None else cases[:measure_n]
    base_rules = list(ALL_RULES)
    lib = [_lp._key(r.lhs, r.rhs) for r in base_rules]
    sink = cost_fn = None
    if measure_fn is None:
        sink = _sink()
        cost_fn = _cost_fn(sink)
    rows = ledger.measure_rows() if ledger is not None else {}
    out: list[LawRecord] = []
    for rank, p in enumerate(proposals):
        armed = armed_keys(p, terms)
        if p.name in rows:
            out.append(LawRecord.from_row(rows[p.name], rank, armed))
            continue
        if measure_fn is not None:
            ev = measure_fn(p, cases, census_op)
        else:
            ev = _pipeline.measure(
                p,
                terms,
                probe_cases,
                base_rules,
                lib,
                census_op,
                sink,
                cost_fn,
                view_oracle=view_oracle,
            )
        rec = LawRecord.from_evidence(p, ev, rank, armed)
        if ledger is not None:
            ledger.append(rec.to_row())
        if verbose:
            print(
                f"[law_evo] {rank + 1}/{len(proposals)} {p.name}: "
                f"shippable={rec.shippable} paid={rec.paid} "
                f"fires={rec.fires} ({rec.no_ship_reason})",
                file=sys.stderr,
                flush=True,
            )
        out.append(rec)
    return out, len(probe_cases)


# ---------------------------------------------------------------------------
#  Fitness + the GA
# ---------------------------------------------------------------------------


def fitness(
    members: Iterable[LawRecord],
    *,
    n_sites: int,
    n_models: int,
    pay_weight: float = 0.5,
) -> tuple[float, float, float]:
    """``(fitness, coverage, pay_rate)`` for a law-set.

    ``coverage`` = |union of member armed sites| / ``n_sites`` — the
    armed-coverage delta the hypothesis names (union, so overlapping
    LHS sites are not double-counted: the *set* is what's scored).
    ``pay_rate`` = Σ member ``paid`` / (|S| · ``n_models``) — the
    mean per-capita probe pay rate.  ``pay_weight`` trades the two;
    the default 0.5 keeps coverage dominant but lets a paying set
    beat a merely-armed one.
    """
    ms = list(members)
    if not ms:
        return 0.0, 0.0, 0.0
    sites: set = set()
    for m in ms:
        sites |= m.armed
    coverage = len(sites) / max(1, n_sites)
    paid = sum(m.paid for m in ms)
    pay_rate = paid / (len(ms) * max(1, n_models))
    return coverage + pay_weight * pay_rate, coverage, pay_rate


def _member_key(member_names: Iterable[str]) -> str:
    """Return the ledger's individual key — sorted names, order-free."""
    return "|".join(sorted(member_names))


def evolve(
    records: list[LawRecord],
    *,
    generations: int = 4,
    pop_size: int = 8,
    set_size: int = 4,
    tournament: int = 3,
    mutation_p: float = 0.3,
    pay_weight: float = 0.5,
    seed: int = 0,
    n_probes: int = 1,
    ledger: EvoLedger | None = None,
) -> dict[str, Any]:
    """Run the evolutionary loop over gauntlet survivors.

    *records* is the measured pool; only ``shippable`` records enter
    the gene pool (the gauntlet filter).  Each individual is a
    frozenset of survivor indices, |S| ≤ *set_size*.  Selection is
    tournament; variation is uniform crossover + per-member
    add/drop/swap mutation; one elite carries over.  Fitness is
    :func:`fitness` — memoized per member-set for the run and
    appended to *ledger* as ``eval`` rows (the expensive half of a
    resume — the ``measure`` rows — is what actually skips work; eval
    rows are the audit trail, fitness itself is cheap enough to
    recompute deterministically).

    ``n_probes`` is the number of probe cases the pay counts were
    measured over (``len(cases)`` in :func:`run_experiment`).

    Returns the run record: per-generation populations, the best
    individual, and the mint-order baseline arm (first *set_size*
    survivors in census rank — the "overnight pass" analogue).
    """
    rng = random.Random(seed)
    survivors = [r for r in records if r.shippable]
    survivors.sort(key=lambda r: r.rank)
    n = len(survivors)
    # Coverage denominator: every armed site any survivor reaches.
    all_sites: set = set()
    for r in survivors:
        all_sites |= r.armed
    n_sites = len(all_sites)

    evals: dict[str, dict[str, Any]] = {}

    def fit_of(idxs: frozenset[int]) -> tuple[float, float, float]:
        names = tuple(sorted(survivors[i].name for i in idxs))
        key = _member_key(names)
        hit = evals.get(key)
        if hit is not None:
            return hit["fitness"], hit["coverage"], hit["pay_rate"]
        f, cov, pay = fitness(
            (survivors[i] for i in idxs),
            n_sites=n_sites or 1,
            n_models=n_probes,
            pay_weight=pay_weight,
        )
        evals[key] = {
            "kind": "eval",
            "members": list(names),
            "fitness": f,
            "coverage": cov,
            "pay_rate": pay,
        }
        return f, cov, pay

    # -- baseline arm: the mint order (first set_size survivors) ----
    baseline_idxs = frozenset(range(min(set_size, n)))
    bfit, bcov, bpay = fit_of(baseline_idxs)

    def random_individual() -> frozenset[int]:
        k = rng.randint(1, set_size)
        return frozenset(rng.sample(range(n), min(k, n)))

    def mutate(ind: frozenset[int]) -> frozenset[int]:
        s = set(ind)
        op = rng.random()
        if op < 0.4 and len(s) > 1:  # drop
            s.discard(rng.choice(tuple(s)))
        elif op < 0.8 and len(s) < set_size:  # add
            choices = [i for i in range(n) if i not in s]
            if choices:
                s.add(rng.choice(choices))
        else:  # swap
            if s:
                s.discard(rng.choice(tuple(s)))
            choices = [i for i in range(n) if i not in s]
            if choices:
                s.add(rng.choice(choices))
        return frozenset(s)

    def crossover(
        a: frozenset[int], b: frozenset[int]
    ) -> frozenset[int]:
        both = a & b
        only_a = tuple(a - b)
        only_b = tuple(b - a)
        child = set(both)
        child.update(
            rng.sample(only_a, min(len(only_a), set_size - len(child)))
        )
        child.update(
            rng.sample(only_b, min(len(only_b), set_size - len(child)))
        )
        while len(child) > set_size:
            child.discard(rng.choice(tuple(child)))
        return frozenset(child) if child else random_individual()

    # -- init: greedy seed + randoms --------------------------------
    singles = sorted(
        ((fit_of(frozenset({i}))[0], i) for i in range(n)),
        reverse=True,
    )
    greedy: list[int] = []
    for _f, i in singles:
        if len(greedy) >= set_size:
            break
        greedy.append(i)
    pop: list[frozenset[int]] = [frozenset(greedy), baseline_idxs]
    while len(pop) < pop_size:
        pop.append(random_individual())

    history: list[dict[str, Any]] = []

    def pick(pool: list[frozenset[int]]) -> frozenset[int]:
        # tournament: best fitness of `tournament` random draws
        best = pool[rng.randrange(len(pool))]
        for _ in range(tournament - 1):
            cand = pool[rng.randrange(len(pool))]
            if fit_of(cand)[0] > fit_of(best)[0]:
                best = cand
        return best

    for gen in range(generations):
        scored = [(fit_of(ind)[0], ind) for ind in pop]
        scored.sort(key=lambda t: -t[0])
        elite = scored[0][1]
        row = {
            "kind": "gen",
            "gen": gen,
            "best": scored[0][0],
            "mean": sum(f for f, _ in scored) / len(scored),
            "pop": [
                {
                    "members": [survivors[i].name for i in sorted(ind)],
                    "fitness": f,
                }
                for f, ind in scored
            ],
        }
        history.append(row)
        if ledger is not None:
            ledger.append(row)
        # breed the next generation
        nxt = [elite]
        while len(nxt) < pop_size:
            child = crossover(pick(pop), pick(pop))
            if rng.random() < mutation_p:
                child = mutate(child)
            nxt.append(child)
        pop = nxt

    # -- final scoring of the last population -----------------------
    final = sorted(
        ((fit_of(ind)[0], ind) for ind in pop), key=lambda t: -t[0]
    )
    best_ind = final[0][1]
    best_members = [survivors[i] for i in sorted(best_ind)]
    bf, bc, bp = fit_of(best_ind)
    if ledger is not None:
        for row in evals.values():
            ledger.append(row)
    bb_names = [survivors[i].name for i in sorted(baseline_idxs)]
    return {
        "n_records": len(records),
        "n_shippable": n,
        "n_sites": n_sites,
        "n_probes": n_probes,
        "n_evals": len(evals),
        "generations": generations,
        "pop_size": pop_size,
        "set_size": set_size,
        "tournament": tournament,
        "mutation_p": mutation_p,
        "pay_weight": pay_weight,
        "seed": seed,
        "baseline": {
            "members": bb_names,
            "fitness": bfit,
            "coverage": bcov,
            "pay_rate": bpay,
        },
        "best": {
            "members": [m.name for m in best_members],
            "fitness": bf,
            "coverage": bc,
            "pay_rate": bp,
            "shippable": all(m.shippable for m in best_members),
            "paid_total": sum(m.paid for m in best_members),
        },
        "history": history,
    }


# ---------------------------------------------------------------------------
#  Ledger — the resumable JSONL record
# ---------------------------------------------------------------------------


class EvoLedger:
    """Append-only JSONL ledger; resume = skip seen measure/eval rows."""

    def __init__(self, path: str | Path) -> None:
        """Open (or create) the ledger at *path*, replaying prior rows."""
        self.path = Path(path)
        self._rows: list[dict] = []
        self._measured: dict[str, dict] = {}
        self._evals: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._rows.append(row)
                if row.get("kind") == "measure":
                    self._measured[row["name"]] = row
                elif row.get("kind") == "eval":
                    key = _member_key(row.get("members", ()))
                    self._evals[key] = row

    def append(self, row: dict) -> None:
        """Append *row* as one JSONL line and index it in memory."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
        self._rows.append(row)
        if row.get("kind") == "measure":
            self._measured[row["name"]] = row
        elif row.get("kind") == "eval":
            self._evals[_member_key(row.get("members", ()))] = row

    def measure_rows(self) -> dict[str, dict]:
        """Return the ``{name: measure-row}`` map the resume path reads."""
        return dict(self._measured)

    def __len__(self) -> int:
        """Return the number of ledger rows seen."""
        return len(self._rows)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    ledger_path: str | Path,
    generations: int = 4,
    pop_size: int = 8,
    set_size: int = 4,
    pool_cap: int = 24,
    n_models: int = 6,
    n_synth: int = 4,
    measure_n: int | None = None,
    pay_weight: float = 0.5,
    seed: int = 0,
    view_oracle: bool = True,
    cases_fn: Callable | None = None,
    measure_fn: Callable | None = None,
    pool_fn: Callable | None = None,
) -> dict[str, Any]:
    """One pilot run: corpus → pool → gauntlet → GA → summary.

    Every knob is injectable so tests drive the whole loop without
    torch; the CLI path uses the real builders.  Returns the report
    dict (also appended to the ledger as a ``summary`` row).
    """
    t0 = time.time()
    ledger = EvoLedger(ledger_path)
    cfn = (
        (lambda: build_cases(n_models, n_synth))
        if cases_fn is None
        else cases_fn
    )
    cases = cfn()
    terms = [c.term for c in cases]
    census_op = census_op_for(cases)

    if pool_fn is not None:
        proposals = pool_fn(census_op, terms)
    else:
        proposals = gen_pool(census_op, terms, cap=pool_cap)

    records, n_probes = measure_pool(
        proposals,
        cases,
        census_op,
        view_oracle=view_oracle,
        measure_n=measure_n,
        ledger=ledger,
        measure_fn=measure_fn,
        verbose=True,
    )
    n_measured = len(records)
    t_measure = time.time()

    result = evolve(
        records,
        generations=generations,
        pop_size=pop_size,
        set_size=set_size,
        pay_weight=pay_weight,
        seed=seed,
        n_probes=n_probes,
        ledger=ledger,
    )
    result.update(
        {
            "kind": "summary",
            "n_cases": len(cases),
            "n_pool": len(proposals),
            "n_measured": n_measured,
            "n_shippable": result["n_shippable"],
            "measure_seconds": round(t_measure - t0, 3),
            "total_seconds": round(time.time() - t0, 3),
            "candidates_per_hour": round(
                n_measured / max(1e-9, (t_measure - t0)) * 3600.0, 1
            ),
            "gauntlet_pass_rate": round(
                result["n_shippable"] / max(1, n_measured), 4
            ),
            "ledger": str(ledger.path),
        }
    )
    ledger.append({k: v for k, v in result.items() if k != "history"})
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m kit_discovery.law_evo --ledger /tmp/exp2.jsonl``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True, help="JSONL ledger path")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--set-size", type=int, default=4)
    ap.add_argument("--pool-cap", type=int, default=24)
    ap.add_argument("--models", type=int, default=6)
    ap.add_argument("--synth", type=int, default=4)
    ap.add_argument(
        "--measure-n",
        type=int,
        default=None,
        help="cap the fire/reach probe subset (default: all cases)",
    )
    ap.add_argument("--pay-weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-view-oracle", action="store_true")
    ap.add_argument("--report", help="write the summary JSON here")
    args = ap.parse_args(argv)

    result = run_experiment(
        ledger_path=args.ledger,
        generations=args.generations,
        pop_size=args.pop,
        set_size=args.set_size,
        pool_cap=args.pool_cap,
        n_models=args.models,
        n_synth=args.synth,
        measure_n=args.measure_n,
        pay_weight=args.pay_weight,
        seed=args.seed,
        view_oracle=not args.no_view_oracle,
    )
    text = json.dumps(result, indent=2, default=str)
    if args.report:
        Path(args.report).write_text(text + "\n")
    # stdout-compat: the headline numbers only.
    print(
        f"pool={result['n_pool']} measured={result['n_measured']} "
        f"shippable={result['n_shippable']} "
        f"pass_rate={result['gauntlet_pass_rate']}"
    )
    print(
        f"best fitness={result['best']['fitness']:.4f} "
        f"members={result['best']['members']}"
    )
    print(
        f"baseline fitness={result['baseline']['fitness']:.4f} "
        f"members={result['baseline']['members']}"
    )
    print(
        f"candidates/hour={result['candidates_per_hour']} "
        f"total={result['total_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
