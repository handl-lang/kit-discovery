"""Moved to ``discoverykit.corpus_cov`` — compatibility shim (plan 0034 phase 1).

``sys.modules`` is swapped so ``kit_discovery.corpus_cov`` *is* the
discoverykit module: attribute access, ``is`` comparisons and
``__module__`` strings behave exactly as before the move.  The
``__main__`` guard re-dispatches ``python -m kit_discovery.corpus_cov``
to the new home; stay-side lazy imports inside the moved code still
spell ``kit_discovery.<stay>`` and resolve here.
"""

import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("discoverykit.corpus_cov", run_name="__main__")
else:
    from discoverykit import corpus_cov as _moved

    _sys.modules[__name__] = _moved
