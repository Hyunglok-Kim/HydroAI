"""
Backward-compatibility shim for the old flat layout.

The library moved to src/hydroai (import hydroai.Data, ...). This shim keeps
legacy sys.path-based imports (import HydroAI.Data as hData, ...) working by
pointing the HydroAI package at the new location, and warns so users migrate.
"""
import os
import warnings

warnings.warn(
    "Importing 'HydroAI.*' via sys.path is deprecated. "
    "Install the package (pip install -e <repo>) and use 'import hydroai.<Module>' instead. "
    "The old import paths will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

# Resolve HydroAI.<Module> against the new package location
__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'hydroai')]
