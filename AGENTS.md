# Agent guide — kit-discovery

The law-discovery engine for the mlopt game — census, proposal,
oracles, evidence store, intake and the `play` domain registry.
Split from `handl-lang/graphopt` `packages/mlopt-discovery` in P3
Step D-1d (the fourth package of the graphopt domain-package
dissolution; the `mlopt-*` → `kit-*` rename).  The package descends
from `tools/law_*.py`; the domain-free mining core was extracted
earlier to `handl-lang/discoverykit` (plan 0034 phase 1), whose
stay-side shims this package still owns.  Python 3.13, torch +
`mlopt-core`/`mlopt-orchestrator`/`mlopt-torch`/`kit-carriers` +
`gamekit`/`discoverykit` on the eqsat/graphopt/game-layer stack.

## Layout

- `src/kit_discovery/` — the one shipped package:
  - proposers: `antiunify`, `combinator`, `form_propose`,
    `forms`, `index_abstract`, `loop_propose`, `state_reindex`,
    `handler_synth`, `family_synth`, `shape_proposal`,
    `learned_propose`, `proposal`/`proposal_core`, `propose`,
    `dspec_synth` (the domain-free halves are discoverykit modules
    reached through the shims);
  - measurement: `census`, `impact`, `oracle`, `verifier`,
    `coherence`/`coherence2`, `lemma_cert`, `pay_gate`, `bounds`,
    `evidence`, `cellstore`, `law_order`, `lawdata/` (the shim onto
    `mlopt_core.laws` tables), `shipped_primitives`,
    `prim_mint`, `macro_moves`, `machine_pack`, `vocab`/`vocabdir`;
  - corpus/intake: `intake/` (`python -m kit_discovery.intake`),
    `model_corpus`, `model_domains`, `modelproj`, `zoo`,
    `workload_gen`, `gap_gen`, `corpus_cov`, `enumerate`,
    `genkernel`, `object_synthesis`, `emit`, `arena`/`arena_player`/
    `meta_arena`/`player_arena`, `families`, `domain_spec`,
    `handl`/`handl_egraph`/`handl_engine` (the handl/egraph bridges);
  - the `.hnl` codec: `hnl_codec`, `hnl_import`, `hnl_surface` —
    the wire contract the `mlopt` product consumes for `.hnl`
    sources/artifacts;
  - games: `open_game`/`open_prior`/`open_self`/`open_verdict`/
    `open_vlabel`, `meta_game`/`meta_player`/`meta_prior`,
    `bilinear_game`, `fragment_game`, `advantage`, `scoring`,
    `training`, `players`, `engine` (shims onto `gamekit`);
  - `play/` — the domain registry: `play/registry.py` is a shim
    onto `discoverykit.play.registry` (`DOMAINS` stays the one
    live dict kernelopt/mlopt write into), `play/domains.py` +
    the per-domain modules (`meta`, `joint`, `search`, `fragment`,
    `gen`, `torch_arm`, …) and `python -m kit_discovery.play`
    (`--domain meta|joint|torch|search|fragment|gen|open|player|bilinear`).
- `tests/` — the package-bound suite (the test files carved out of
  graphopt's `tests/` whose imports stay inside this package's dep
  closure: `kit_discovery` + `mlopt_core`/`mlopt_orchestrator`/
  `mlopt_torch`/`kit_carriers`/`mlopt_dtypes` + `gamekit`/
  `discoverykit` + torch).  Cross-stack tests (anything touching
  `bench.*`, `mlopt_host_tensor`, `mlopt_handl_peer`, `kit_onnx`,
  `kit_cuda`, the e2e model benches) stayed in graphopt and import
  `kit_discovery` as a git dep.

## Contracts

- **Top of the stack** — `kit_discovery` reads everything below it
  (`mlopt_torch`, `mlopt_orchestrator`, `mlopt_core`,
  `kit_carriers`, `gamekit`, `discoverykit`) and nothing may depend
  back on it; the import-linter `forbidden` contract enforces the
  upward edges (`mlopt_host_tensor`, `kit_onnx`, `benchkit`,
  `mlopt_native`, the `mlopt` product).  `kit_cuda` is a lazy
  optional bind (`ImportError` → honest decline in `play/gen.py` /
  `play/torch_arm.py`) — deliberately undeclared.
- The `sys.modules` shim modules (`engine`, `scoring`, `advantage`,
  `players`, `play/registry`, `lawdata`, …) swap onto the
  `gamekit`/`discoverykit`/`mlopt_core` targets — `.pyi` stubs
  carry the re-export surface for `ty`; do not "fix" them into
  real modules.
- Substrate deps: `mlopt-core`, `mlopt-orchestrator`,
  `mlopt-dtypes` (handl-lang/eqsat, `packages/` subdirs,
  tag-pinned); `mlopt-torch` (handl-lang/graphopt,
  `packages/mlopt-torch` subdir — resolved by the *consumer's*
  `[tool.uv.sources]`; this repo's dev env binds it via a
  dev-group direct-URL, the kit-carriers dep-source lesson);
  `kit-carriers` (handl-lang/kit-carriers, tag-pinned);
  `gamekit`/`discoverykit` (their repos, `packages/` subdirs,
  tag-pinned).  `discoverykit` lazily imports `kit_discovery.*`
  stay-side names at call time — undeclared by design, resolved
  through the consumer's env.
- `REPO_ROOT` (`__init__.py`) is this repo's root — the
  `emit.py` tensor-law patch path and the `.venv/bin/ruff` probe
  resolve against it; the `hnl-run`/`egraph-run` bridges resolve
  through `MLOPT_HNL_RUN`/`MLOPT_EGRAPH_RUN` env overrides and
  `~/.cargo/shared-target` fallbacks.
- History was preserved via `git subtree split` of
  `graphopt:packages/mlopt-discovery` — `git log` here reaches back
  through the graphopt history.

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src
.venv/bin/ruff format --check src
.venv/bin/ty check
.venv/bin/lint-imports
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

- Coverage floor is `fail_under` in `[tool.coverage.report]`,
  measured on the package-bound suite — the cross-stack exercise
  still runs through graphopt's suite via the git dep.  Tighten,
  never lower.
- The `requires_cuda` marker skips cleanly at setup when CUDA is
  absent or OOMs on a probe allocation (shared `tests/conftest.py`).

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
