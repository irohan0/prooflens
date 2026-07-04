"""Global determinism: seed Python / NumPy / Torch and set deterministic flags where feasible.

The seed is logged in every results provenance header (docs/TROUBLESHOOTING.md §Reproducibility).
NumPy and Torch are seeded only if importable, so this module stays usable in the light local
environment (metrics/loaders) that has neither installed yet.
"""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed all RNGs we might touch. Idempotent; safe to call before heavy imports exist."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism; some CUDA kernels remain nondeterministic and that is fine
        # for retrieval scoring, which is dominated by the (deterministic) index search.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
