"""Moved to ``discoverykit.meta_prior`` — compatibility shim (plan 0034 phase 1).

``sys.modules`` is swapped so ``kit_discovery.meta_prior`` *is* the
discoverykit module: attribute access, ``is`` comparisons and
``__module__`` strings behave exactly as before the move.  The
``__main__`` guard re-dispatches ``python -m kit_discovery.meta_prior``
to the new home; stay-side lazy imports inside the moved code still
spell ``kit_discovery.<stay>`` and resolve here.
"""

import runpy as _runpy
import sys as _sys

if __name__ == "__main__":
    _runpy.run_module("discoverykit.meta_prior", run_name="__main__")
else:
    from discoverykit import meta_prior as _moved

    _sys.modules[__name__] = _moved
