import hashlib
import random
from typing import Any

import numpy as np
import torch

try:
    import open3d as o3d
except Exception:
    o3d = None


def normalize_seed(random_seed: int) -> int:
    # Keep all libraries on the same deterministic seed while staying inside
    # the signed 32-bit range accepted by Open3D's pybind binding.
    return int(int(random_seed) % (2**31 - 1))


def set_seed(random_seed: int) -> None:
    normalized_seed = normalize_seed(random_seed)
    np.random.seed(normalized_seed)
    random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    torch.cuda.manual_seed_all(normalized_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if o3d is not None and hasattr(o3d.utility, "random") and hasattr(o3d.utility.random, "seed"):
        o3d.utility.random.seed(normalized_seed)


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "::".join([str(int(base_seed)), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)
