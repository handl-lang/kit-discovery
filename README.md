# kit-discovery

The law-discovery engine for the mlopt game — census, proposal,
oracles, evidence store, intake and the `play` domain registry.

`kit_discovery` is the **top of the torch stack**: it reads
`mlopt_core`/`mlopt_orchestrator` (handl-lang/eqsat),
`kit_torch`/`kit_carriers` (handl-lang/graphopt /
handl-lang/kit-carriers) and the game layer (`gamekit` +
`discoverykit`) — nothing may depend back on it.  `kit_cuda` is a
lazy optional bind (CUDA play arms decline honestly when it is
absent).

Headline modules: `antiunify`/`combinator`/`form_propose`/
`index_abstract`/`loop_propose`/`handler_synth` (the proposers —
their domain-free halves live in discoverykit), `census`/`impact`/
`oracle`/`verifier`/`evidence`/`pay_gate` (measurement),
`intake/` + `model_corpus`/`modelproj` (corpus intake),
`hnl_codec`/`hnl_import`/`hnl_surface` (the `.hnl` wire codec the
mlopt product consumes), `play/` (the `DOMAINS` registry +
`python -m kit_discovery.play` CLI), and the `mlopt_discovery.*`
compat shims that swap `sys.modules` onto `gamekit`/`discoverykit`.

## Develop

```sh
uv sync --group dev
```

or as a git dependency:

```toml
kit-discovery = { git = "https://github.com/handl-lang/kit-discovery.git", tag = "v0.1.0" }
```

Substrate pins (`mlopt-core`, `mlopt-orchestrator`, `mlopt-dtypes`,
`gamekit`, `discoverykit`, `kit-carriers`) resolve from their own
repos — see `[tool.uv.sources]`; `kit-torch` resolves from
`handl-lang/graphopt` through the consumer's sources.

## Verify

```sh
uv run scripts/gate.py --coverage   # pytest → ruff → ty → lint-imports → coverage floor
```

`scripts/gate.py` is a stdlib runner over the dev toolchain; `workflows/gate.yaml`
is the gate entry.

## License

MIT.
