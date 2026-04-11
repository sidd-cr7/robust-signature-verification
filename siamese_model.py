"""Compatibility shim for legacy imports.

Some scripts (including verification/test_signature.py) expect a module named
`siamese_model` that exports `SiameseNetwork`. The real implementation lives in
`verification/siamese_train.py`.

This file makes `from siamese_model import SiameseNetwork` work without
requiring changes to existing scripts.
"""

from verification.siamese_train import SiameseNetwork

__all__ = ["SiameseNetwork"]
