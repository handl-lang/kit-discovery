"""mlopt term ↔ serde-shaped handl ``Term`` JSON — the wire codec.

This is the JSON half of the ``.hnl`` codec pair (ADR-0172): where
:mod:`.hnl_surface` / :mod:`.hnl_import` move terms through *surface
syntax*, this module moves them through the **Term-JSON** envelope —
the externally-tagged serde shape ``hnl-engine`` parses
(``{"Op": …}`` / ``{"Arr": …}`` / ``{"Const": …}`` / ``{"Opaque": …}``).

**ADR-0178 E-3: this module is now a shim.**  The codec itself moved
to ``mlopt_core.term_json`` (eqsat v0.1.12) — ONE kernel-Term-JSON
codec for the whole stack, reachable from ``mlopt-orchestrator``
(which cannot import ``kit_discovery`` — the dep graph forbids it).
The historical private implementation (479 lines, ``{"Foreign":
"tensor"}`` source + ``lit.__mlopt_attrs__`` attrs + nested-record
dag ``dim``) is deleted; the public names below are re-exports.

Emit profile — the **canonical** spelling, which is what this
surface's consumers actually need:

* ``source: "Native"`` — load-bearing for law matching (patterns pin
  ``source:"native"`` under exact-keyset attr match) and accepted
  everywhere the old ``{"Foreign": "tensor"}`` was: decode ignores
  ``source`` entirely, ``source-onnx`` emits its own spelling, and
  the ``onnx_lifter_differential`` canonicalizer never reads it.
* attrs under the canonical ``attrs`` field — the ADR-0173 forward
  spelling ``source-onnx`` already emits; ``lit.__mlopt_attrs__``
  remains decode-legacy.
* ``Var``/``Param`` leaves as structured ``Arr`` records — the
  spelling ``source-onnx`` and the differential's leaf census
  require (a ``Const``-string leaf would compare as a different node
  kind).
* ``dag.program`` records carry ``Op.dim`` as a *node-table index*
  (the engine-eqsat ``dag.rs`` schema) — strictly more compatible
  than the old nested-record ``dim``, which engine-eqsat refuses.

Decode is permanently tolerant: both leaf spellings (canonical
``Arr`` record and the legacy ``__mlopt_leaf__:-prefixed ``Const``
string), both attrs channels, any ``ArrSource`` spelling, and
nested-record dag ``dim`` all read identically — old records keep
working.

Honest declines are unchanged in spirit: ``dag=False`` refuses a
shared term, ``allow_dag=False`` refuses the record, unknown
``Opaque`` tags and malformed node tables raise
:class:`HnlCodecError` (now an alias of
``mlopt_core.term_json.TermJsonError``).
"""

from __future__ import annotations

from mlopt_core.term_json import (
    TermJsonError as HnlCodecError,
)
from mlopt_core.term_json import (
    term_from_json,
    term_to_json,
)

__all__ = [
    "HnlCodecError",
    "term_from_json",
    "term_to_json",
]
